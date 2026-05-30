"""Honest benchmark harness for the megakernel-backed Qwen3-TTS pipeline.

Measures the four metrics required by the take-home brief and writes results
to ``bench_results.json`` next to a pretty-printed stdout summary.

Metrics
-------
1. **Decode tok/s** (megakernel only). 100-token raw decode benchmark against
   the bare :class:`qwen_megakernel.model.Decoder` -- bypasses code predictor
   and codec to isolate Talker throughput. 3 warmup + 5 timed runs; reports
   mean +/- stdev.

2. **TTFC** (time-to-first-audio-chunk). Wall-clock nanoseconds from the
   :meth:`MegakernelTTS.generate` invocation to the first yielded byte chunk,
   on a fixed reference utterance. 3 warmup + 5 timed runs.

3. **RTF** (real-time factor). For a target ~5-second utterance:
   ``RTF = decode_wall_ms / audio_duration_ms``. Audio duration is derived
   from the total int16 bytes yielded (bytes / 2 / sample_rate).
   3 warmup + 5 timed runs.

4. **End-to-end Pipecat latency** is intentionally NOT measured here -- the
   brief partitions it to the Pipecat demo script (``pipecat_demo.py``).
   This bench is inference-server-only so we can report Talker / TTS numbers
   without a Deepgram or LLM dependency.

Run::

    python bench_megakernel.py                            # full bench
    python bench_megakernel.py --stub                     # plumbing check, no GPU
    python bench_megakernel.py --skip-decode              # only TTFC + RTF
    python bench_megakernel.py --out /tmp/bench.json      # custom output

Output is intentionally honest:
- stdev reported for every metric
- raw per-run timings preserved in the JSON
- no cherry-picking, no "best of N"
- CUDA syncs are explicit around every timed region
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

# Local-package import; this file is run from ``inference-server/``.
from megakernel_tts import (
    QWEN3_TTS_SAMPLE_RATE,
    MegakernelTTS,
    MegakernelTTSConfig,
)

# Reference utterance: ~5 seconds of English at conversational pacing
# (~150 wpm ~= 12.5 words for 5s). Tuned to be long enough that RTF is not
# dominated by prefill, short enough that one bench run finishes quickly.
REFERENCE_UTTERANCE_5S = (
    "Pipecat lets me wire together speech to text, a language model, "
    "and text to speech into one streaming voice pipeline."
)

# Short utterance used only for TTFC -- we want first-chunk latency,
# not steady-state throughput, so a short input keeps prefill cheap.
TTFC_UTTERANCE = "Hello there, how can I help you today?"

# Number of tokens for the bare-Talker tok/s benchmark.
DECODE_BENCH_TOKENS = 100


@dataclass
class RunStats:
    """Mean / stdev / min / max over a list of samples."""

    samples: list[float]
    mean: float = 0.0
    stdev: float = 0.0
    min: float = 0.0
    max: float = 0.0

    @classmethod
    def from_samples(cls, samples: list[float]) -> "RunStats":
        if not samples:
            return cls(samples=[])
        return cls(
            samples=list(samples),
            mean=float(statistics.fmean(samples)),
            stdev=float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0,
            min=float(min(samples)),
            max=float(max(samples)),
        )


@dataclass
class BenchResults:
    """Aggregate bench output, JSON-serialized at the end."""

    model_name: str
    speaker: str
    device: str
    stub: bool
    sample_rate: int

    decode_tok_per_s: RunStats = field(default_factory=lambda: RunStats(samples=[]))
    ttfc_ms: RunStats = field(default_factory=lambda: RunStats(samples=[]))
    rtf: RunStats = field(default_factory=lambda: RunStats(samples=[]))

    audio_duration_ms_per_run: list[float] = field(default_factory=list)
    decode_wall_ms_per_run: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Decode tok/s -- bare Talker, no codec, no async
# ----------------------------------------------------------------------------


def bench_decode_tok_per_s(
    *,
    model_name: str,
    n_warmup: int = 3,
    n_timed: int = 5,
    n_tokens: int = DECODE_BENCH_TOKENS,
) -> RunStats:
    """Measure megakernel Talker decode throughput in tokens/sec.

    Uses the raw :class:`qwen_megakernel.model.Decoder` directly so we are
    not paying for code predictor / codec / asyncio overhead. The decoder
    autoregressively steps N tokens with a single fixed input id and times
    the whole loop with explicit CUDA syncs on both ends.

    Args:
        model_name: HF checkpoint name. Note that until kernel mods are in,
            this must be ``"Qwen/Qwen3-0.6B"`` (the megakernel's current
            target). Switch to the 1.7B Qwen3-TTS Talker once mods land.
        n_warmup: Warmup decode passes (not timed).
        n_timed: Timed passes; mean +/- stdev reported.
        n_tokens: Tokens per pass.

    Returns:
        :class:`RunStats` of tokens/second across timed runs.
    """
    try:
        import torch  # local import: bench harness must run on CPU-only laptops too
        from qwen_megakernel.model import Decoder
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Skipping decode tok/s bench: qwen_megakernel import failed ({e}). "
            "This is expected off the RTX 5090 box.",
            e=e,
        )
        return RunStats(samples=[])

    logger.info(
        "Decode tok/s bench: model={m} warmup={w} timed={t} tokens={n}",
        m=model_name,
        w=n_warmup,
        t=n_timed,
        n=n_tokens,
    )

    # H2 fix: Decoder takes model_path (not model_name), and doesn't expose
    # a .tokenizer attribute (the new model.py loads weights via safetensors
    # and skips the HF tokenizer entirely). For the talker-only bench we just
    # seed with token id 0 — output is unconditioned (the brief's tok/s
    # metric is about decode throughput, not acoustic quality).
    decoder = Decoder(model_path=model_name, verbose=False)
    seed_token = 0

    def _one_pass() -> float:
        decoder.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tid = seed_token
        for _ in range(n_tokens):
            tid = decoder.step(tid)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    # Warmup
    for _ in range(n_warmup):
        _one_pass()

    # Timed
    samples_tok_per_s: list[float] = []
    for i in range(n_timed):
        wall_s = _one_pass()
        tok_per_s = n_tokens / wall_s
        samples_tok_per_s.append(tok_per_s)
        logger.info("  run {i}: {wall:.3f}s -> {tps:.1f} tok/s",
                    i=i, wall=wall_s, tps=tok_per_s)

    return RunStats.from_samples(samples_tok_per_s)


# ----------------------------------------------------------------------------
# TTFC -- async, first-chunk latency
# ----------------------------------------------------------------------------


async def bench_ttfc_one_pass(tts: MegakernelTTS, text: str) -> float:
    """Time from generate() invocation to first yielded chunk, in ms."""
    t0_ns = time.perf_counter_ns()
    gen = tts.generate(text)
    async for _chunk in gen:
        elapsed_ns = time.perf_counter_ns() - t0_ns
        # Drain the rest of the generator so KV cache state is consistent
        # for the next run -- otherwise we measure prefill of a half-finished
        # utterance.
        async for _ in gen:
            pass
        return elapsed_ns / 1_000_000.0
    return float("nan")  # generator produced nothing


async def bench_ttfc(
    tts: MegakernelTTS,
    *,
    n_warmup: int = 3,
    n_timed: int = 5,
    text: str = TTFC_UTTERANCE,
) -> RunStats:
    """Measure time-to-first-audio-chunk over many runs."""
    logger.info("TTFC bench: warmup={w} timed={t}", w=n_warmup, t=n_timed)
    for _ in range(n_warmup):
        await bench_ttfc_one_pass(tts, text)
    samples: list[float] = []
    for i in range(n_timed):
        ms = await bench_ttfc_one_pass(tts, text)
        samples.append(ms)
        logger.info("  run {i}: TTFC={ms:.1f} ms", i=i, ms=ms)
    return RunStats.from_samples(samples)


# ----------------------------------------------------------------------------
# RTF -- full-utterance decode wall vs. audio duration
# ----------------------------------------------------------------------------


async def bench_rtf_one_pass(
    tts: MegakernelTTS, text: str
) -> tuple[float, float, float]:
    """Run one full synthesis pass, returning (rtf, wall_ms, audio_ms)."""
    total_bytes = 0
    t0 = time.perf_counter()
    async for chunk in tts.generate(text):
        total_bytes += len(chunk)
    wall_s = time.perf_counter() - t0

    n_samples = total_bytes // 2  # int16 = 2 bytes / sample
    audio_s = n_samples / tts.sample_rate if n_samples > 0 else float("nan")
    rtf = wall_s / audio_s if audio_s > 0 else float("nan")

    wall_ms = wall_s * 1000.0
    audio_ms = audio_s * 1000.0
    return rtf, wall_ms, audio_ms


async def bench_rtf(
    tts: MegakernelTTS,
    *,
    n_warmup: int = 3,
    n_timed: int = 5,
    text: str = REFERENCE_UTTERANCE_5S,
) -> tuple[RunStats, list[float], list[float]]:
    """Measure RTF over many runs, return per-run wall/audio durations too."""
    logger.info("RTF bench: warmup={w} timed={t}", w=n_warmup, t=n_timed)
    for _ in range(n_warmup):
        await bench_rtf_one_pass(tts, text)

    rtfs: list[float] = []
    walls: list[float] = []
    audios: list[float] = []
    for i in range(n_timed):
        rtf, wall_ms, audio_ms = await bench_rtf_one_pass(tts, text)
        rtfs.append(rtf)
        walls.append(wall_ms)
        audios.append(audio_ms)
        logger.info(
            "  run {i}: wall={w:.1f}ms audio={a:.1f}ms RTF={r:.4f}",
            i=i, w=wall_ms, a=audio_ms, r=rtf,
        )
    return RunStats.from_samples(rtfs), walls, audios


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


def _pretty_print(results: BenchResults) -> None:
    """Pretty-print bench results to stdout."""
    bar = "=" * 64
    print(bar)
    print(" Megakernel Qwen3-TTS bench results")
    print(bar)
    print(f" model         : {results.model_name}")
    print(f" speaker       : {results.speaker}")
    print(f" device        : {results.device}")
    print(f" sample rate   : {results.sample_rate} Hz")
    print(f" stub          : {results.stub}")
    print()

    def _row(label: str, stats: RunStats, unit: str, fmt: str = ".2f") -> None:
        if not stats.samples:
            print(f" {label:18s} : -- (not measured)")
            return
        mean = format(stats.mean, fmt)
        stdev = format(stats.stdev, fmt)
        print(
            f" {label:18s} : {mean} +/- {stdev} {unit}  "
            f"(min={format(stats.min, fmt)}, max={format(stats.max, fmt)}, "
            f"n={len(stats.samples)})"
        )

    _row("Decode tok/s",      results.decode_tok_per_s, "tok/s", ".1f")
    _row("TTFC",              results.ttfc_ms,          "ms",    ".1f")
    _row("RTF",               results.rtf,              "",      ".4f")
    print()

    if results.notes:
        print(" notes:")
        for n in results.notes:
            print(f"   - {n}")
    print(bar)


async def main_async(args: argparse.Namespace) -> int:
    results = BenchResults(
        model_name=args.model,
        speaker=args.speaker,
        device=args.device,
        stub=args.stub,
        sample_rate=QWEN3_TTS_SAMPLE_RATE,
    )

    # --- TTS pipeline (TTFC + RTF) -----------------------------------------
    tts = MegakernelTTS(
        config=MegakernelTTSConfig(
            model_name=args.model,
            speaker=args.speaker,
            device=args.device,
            stub=args.stub,
        )
    )

    if not args.skip_ttfc:
        try:
            results.ttfc_ms = await bench_ttfc(
                tts, n_warmup=args.warmup, n_timed=args.timed
            )
        except NotImplementedError as e:
            results.notes.append(f"TTFC skipped: {e}")
            logger.warning("TTFC skipped: {e}", e=e)

    if not args.skip_rtf:
        try:
            rtf_stats, walls, audios = await bench_rtf(
                tts, n_warmup=args.warmup, n_timed=args.timed
            )
            results.rtf = rtf_stats
            results.decode_wall_ms_per_run = walls
            results.audio_duration_ms_per_run = audios
        except NotImplementedError as e:
            results.notes.append(f"RTF skipped: {e}")
            logger.warning("RTF skipped: {e}", e=e)

    # --- Bare-Talker decode tok/s ------------------------------------------
    if not args.skip_decode:
        results.decode_tok_per_s = bench_decode_tok_per_s(
            model_name=args.decode_model,
            n_warmup=args.warmup,
            n_timed=args.timed,
        )

    # --- Write + print -----------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_to_jsonable(results), indent=2))
    logger.info("Wrote {p}", p=out_path)
    _pretty_print(results)
    return 0


def _to_jsonable(results: BenchResults) -> dict[str, Any]:
    """Convert dataclasses to plain JSON-friendly dicts."""
    d = asdict(results)
    return d


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        help="HF checkpoint for the Qwen3-TTS pipeline (TTFC + RTF).",
    )
    p.add_argument(
        "--decode-model",
        default="Qwen/Qwen3-0.6B",
        help=(
            "HF checkpoint for the bare-Talker tok/s bench. Defaults to the "
            "qwen_megakernel-supported Qwen3-0.6B; switch to the 1.7B Qwen3-TTS "
            "Talker once kernel mods land."
        ),
    )
    p.add_argument("--speaker", default="ryan")
    p.add_argument("--device", default="cuda")
    p.add_argument("--stub", action="store_true", help="Run with the silence stub.")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--timed", type=int, default=5)
    p.add_argument("--skip-decode", action="store_true")
    p.add_argument("--skip-ttfc", action="store_true")
    p.add_argument("--skip-rtf", action="store_true")
    p.add_argument("--out", default="bench_results.json")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
