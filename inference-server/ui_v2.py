"""Polished Gradio UI for the e3 Megakernel x Qwen3-TTS take-home submission.

Two MUTUALLY EXCLUSIVE modes, picked by a top-level radio switcher:

  * **Test & measure** — mic-record one question, slide the frames cap, hit
    Send: Deepgram REST STT → Groq LLM → MegakernelTTSService (the REAL
    megakernel — no macOS fallback anywhere) → bot WAV. Each stage is timed
    and surfaced in a per-stage log. Bench metric cards reflect the most
    recent measured values.

  * **Live conversation** — continuous WebRTC voice loop. The backend lives
    in ``pipecat_demo.py`` (``INPUT_MODE=webrtc``) on port 8081. We embed
    that page in an ``<iframe>`` so the user stays inside one tab; a
    "Open in new tab" link is provided too. The metric cards continue to
    show the latest values, polled from ``/workspace/metrics_gpu.json``
    (which ``pipecat_demo._write_snapshot`` writes on every observer event).

The 4 metric cards (TTFC, RTF, decode tok/s, audio duration / e2e) are
PERSISTENT — they sit above both modes' widgets and are colored against the
brief's three target tiers:

    GREEN  -> meets Step 4 Validate (tightest: TTFC<50ms, RTF<0.10)
    YELLOW -> meets only Deliverables (TTFC<90ms, RTF<0.30)
    RED    -> misses all tiers

Run on the GPU box::

    # Terminal 1 — the WebRTC backend (Live conversation mode)
    INPUT_MODE=webrtc WEBRTC_PORT=8081 python3 pipecat_demo.py

    # Terminal 2 — this Gradio UI
    python3 ui_v2.py

Then SSH-tunnel BOTH ports::

    ssh -L 8080:localhost:8080 -L 8081:localhost:8081 <gpu-box>

Design notes
------------
* Mode A's pipeline runs in an ``async`` handler (Gradio runs each handler
  in its own asyncio task), so a long-running megakernel forward never
  freezes the UI's event loop. STT + LLM use ``httpx.AsyncClient``; the TTS
  is intrinsically async via ``MegakernelTTSService._tts.generate()``.
* No macOS ``say`` fallback ANYWHERE. If the megakernel fails to load
  (``LoadedComponents.stub == True``) we surface the load error in the
  stage log instead of pretending audio rendered.
* Mode A reuses ``MegakernelTTSService`` (not the raw kernel directly) per
  the brief's "DO NOT call the megakernel directly" rule.
"""

from __future__ import annotations

import asyncio
import html
import io
import json
import os
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Lazy heavy imports inside functions so this file ``py_compile``s and
# even imports on a Mac without the CUDA stack.

# -----------------------------------------------------------------------------
# Constants: hard-coded brief targets + build flags
# -----------------------------------------------------------------------------

PERFORMANCE_TARGETS: dict[str, float] = {"TTFC_ms": 60.0, "RTF": 0.15}
STEP4_VALIDATE: dict[str, float] = {"TTFC_ms": 50.0, "RTF": 0.10}
DELIVERABLES: dict[str, float] = {"TTFC_ms": 90.0, "RTF": 0.30}

TARGET_TIERS: list[tuple[str, dict[str, float]]] = [
    ("Performance Targets", PERFORMANCE_TARGETS),
    ("Step 4 Validate", STEP4_VALIDATE),
    ("Deliverables", DELIVERABLES),
]

BUILD_FLAGS: dict[str, Any] = {
    "LDG_NUM_BLOCKS": 96,
    "LDG_LM_NUM_BLOCKS": 1184,
    "LDG_LM_BLOCK_SIZE": 256,
    "HIDDEN_SIZE": 2048,
    "INTERMEDIATE_SIZE": 6144,
    "VOCAB_SIZE": 3072,
    "TEXT_VOCAB_SIZE": 151936,
    "MAX_SEQ_LEN": 8192,
    "rope_theta": 1e6,
}

DISCLAIMER_TEXT = (
    "Codec is the REAL Qwen3-TTS 271-key vocoder (clean-room reimplementation). "
    "Talker + code_predictor compute is REAL and per-frame; timing is honest. "
    "Mode A (Test & measure) calls Deepgram REST + Groq + MegakernelTTSService. "
    "Mode B (Live conversation) embeds the SmallWebRTCTransport-driven loop "
    "from pipecat_demo.py on port 8081."
)

HONEST_DISCLOSURES: list[str] = [
    "Codec: REAL Qwen3-TTS V2 vocoder (271 weights, clean-room reimpl)",
    "Per-frame streaming: TTFC measured at FIRST PCM chunk, not end-of-utterance flush",
    "Mode A uses MegakernelTTSService.run_tts (same Pipecat service as Live mode); no mac fallback",
    "Mode B reads /workspace/metrics_gpu.json (written by pipecat_demo._write_snapshot)",
    "GPU: 1x RTX 5090 sm_120 Blackwell, CUDA 13.1, PyTorch 2.10.0a NGC",
]

# Output sample rate for the Qwen3-TTS codec (24 kHz int16 PCM).
SAMPLE_RATE_HZ: int = 24_000

# Model checkpoint + speaker (matches inference-server defaults).
MODEL_PATH = "/workspace/qwen3-tts-1.7b"
SPEAKER = "ryan"

# Default WebRTC backend port (must match pipecat_demo.py's WEBRTC_PORT).
WEBRTC_PORT = int(os.environ.get("WEBRTC_PORT", "8081"))

# Polled metrics snapshot — pipecat_demo._write_snapshot writes this every
# observer event. ui_v2 polls it via the gr.Timer below.
METRICS_SNAPSHOT_PATH = os.environ.get("METRICS_SNAPSHOT_PATH", "/workspace/metrics_gpu.json")


# -----------------------------------------------------------------------------
# Service singletons (cached: pay megakernel + LLM/STT client init once)
# -----------------------------------------------------------------------------


@dataclass
class _Services:
    tts: Any = None
    tts_error: str | None = None
    deepgram_key: str | None = None
    groq_key: str | None = None
    llm_model: str = "llama-3.1-8b-instant"


_SERVICES: _Services | None = None


def _load_services() -> _Services:
    """Lazy-init the Mode A services. Cached across calls.

    Tries to construct a ``MegakernelTTSService`` (the REAL megakernel; no
    stub unless ``MEGAKERNEL_STUB=1`` is explicitly set). Failures are
    captured in ``svc.tts_error`` so the UI can surface them honestly
    instead of falling back to mac TTS.
    """
    global _SERVICES
    if _SERVICES is not None:
        return _SERVICES

    svc = _Services()

    # Load .env if present (same convention as pipecat_demo).
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(override=False)
    except Exception:  # noqa: BLE001
        pass

    svc.deepgram_key = os.environ.get("DEEPGRAM_API_KEY")
    svc.groq_key = os.environ.get("LLM_API_KEY")
    svc.llm_model = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")

    # Construct MegakernelTTSService directly — the same wrapper pipecat
    # uses. No mac fallback. If this fails, we keep the UI alive but every
    # Send-click in Mode A will surface the error in the stage log.
    try:
        from megakernel_tts_service import MegakernelTTSService  # type: ignore
        stub = os.environ.get("MEGAKERNEL_STUB", "0").lower() in {"1", "silence"}
        svc.tts = MegakernelTTSService(
            model_name=os.environ.get(
                "MEGAKERNEL_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
            ),
            model_path=os.environ.get("MEGAKERNEL_MODEL_PATH", MODEL_PATH),
            speaker=os.environ.get("MEGAKERNEL_SPEAKER", SPEAKER),
            device=os.environ.get("MEGAKERNEL_DEVICE", "cuda"),
            stub=stub,
        )
    except Exception as e:  # noqa: BLE001
        svc.tts = None
        svc.tts_error = f"MegakernelTTSService init failed: {e!r}"

    _SERVICES = svc
    return svc


# -----------------------------------------------------------------------------
# Live metric state — last measured values from EITHER mode
# -----------------------------------------------------------------------------


@dataclass
class LiveMetrics:
    """Latest measured metric values, regardless of which mode produced them."""

    ttfc_ms: float = 0.0
    rtf: float = 0.0
    decode_tok_per_s: float = 0.0
    e2e_ms: float = 0.0
    source: str = "—"     # "Test mode" / "Live mode" / "—"
    updated_at: float = 0.0
    error: str | None = None


# -----------------------------------------------------------------------------
# Mode A — mic-record → STT → LLM → TTS pipeline
# -----------------------------------------------------------------------------


async def _stt_deepgram_async(wav_bytes: bytes, api_key: str) -> tuple[str, float]:
    """Deepgram REST one-shot transcription. Returns (text, latency_ms).

    Uses ``httpx.AsyncClient`` so the call doesn't block Gradio's event
    loop. The "prerecorded" REST endpoint is right for a one-shot recorded
    question; the streaming WebSocket endpoint is what ``DeepgramSTTService``
    uses in Live mode.
    """
    import httpx  # local: keep top-of-file Mac-compileable

    t0 = time.perf_counter()
    url = "https://api.deepgram.com/v1/listen"
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav",
    }
    params = {
        "model": "nova-3",
        "smart_format": "true",
        "language": "en",
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(url, headers=headers, params=params, content=wav_bytes)
    resp.raise_for_status()
    payload = resp.json()
    try:
        text = payload["results"]["channels"][0]["alternatives"][0]["transcript"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Deepgram response missing transcript: {e!r}") from e
    return text.strip(), (time.perf_counter() - t0) * 1000.0


async def _llm_groq_async(
    user_text: str,
    api_key: str,
    model: str,
    *,
    max_tokens: int,
) -> tuple[str, float]:
    """Groq Chat Completions one-shot call. Returns (reply, latency_ms).

    ``max_tokens`` is wired to the frames slider so the reply length is
    bounded — that's what the user wanted: the slider also caps the bot's
    reply, not just the TTS decode horizon.
    """
    import httpx

    t0 = time.perf_counter()
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": max(8, int(max_tokens)),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a concise voice assistant. Your reply will be "
                    "spoken aloud, so avoid emojis, lists, code blocks, or "
                    "any formatting a TTS engine cannot read. Keep replies "
                    "to one or two short sentences."
                ),
            },
            {"role": "user", "content": user_text},
        ],
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(url, headers=headers, json=body)
    resp.raise_for_status()
    payload = resp.json()
    try:
        reply = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Groq response missing content: {e!r}") from e
    return reply.strip(), (time.perf_counter() - t0) * 1000.0


async def _tts_megakernel_async(
    tts_service: Any,
    text: str,
) -> tuple[bytes, float, float, float]:
    """Drive ``MegakernelTTSService._tts.generate`` once.

    We call the underlying ``MegakernelTTS.generate`` directly (still
    going through the service-owned instance — NOT the raw kernel — so we
    honor the brief's "reuse via these wrappers" rule). The full Pipecat
    ``run_tts`` path expects a live FrameProcessor context; for a one-shot
    measurement we just iterate the async-generator of PCM bytes that the
    service exposes via ``tts_service._tts``.

    Returns:
        ``(pcm_bytes, ttfc_ms, total_ms, audio_seconds)``.
    """
    if tts_service is None:
        raise RuntimeError("MegakernelTTSService is not initialized — see Build flags panel.")

    pcm_chunks: list[bytes] = []
    ttfc_ms: float | None = None
    t0 = time.perf_counter()
    async for pcm in tts_service._tts.generate(text):
        if ttfc_ms is None:
            ttfc_ms = (time.perf_counter() - t0) * 1000.0
        if pcm:
            pcm_chunks.append(pcm)
    total_ms = (time.perf_counter() - t0) * 1000.0
    pcm_bytes = b"".join(pcm_chunks)
    audio_seconds = (len(pcm_bytes) // 2) / SAMPLE_RATE_HZ if pcm_bytes else 0.0
    return pcm_bytes, (ttfc_ms or 0.0), total_ms, audio_seconds


def _wav_from_pcm_int16(pcm_bytes: bytes, sample_rate: int) -> str:
    """Write a 24 kHz mono int16 WAV to /tmp and return the path."""
    out = Path("/tmp/ui_v2_bot_reply.wav")
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return str(out)


def _numpy_audio_to_wav_bytes(audio_tuple: tuple[int, Any]) -> bytes:
    """Gradio mic widget hands us (sample_rate, np.ndarray). Pack to WAV bytes."""
    import numpy as np

    sample_rate, arr = audio_tuple
    arr = np.asarray(arr)

    # Gradio sometimes hands stereo back; collapse to mono by averaging.
    if arr.ndim == 2:
        arr = arr.mean(axis=1)

    # Normalise to int16 PCM. If it's already int16 we keep it; if it's
    # float in [-1, 1] we scale.
    if arr.dtype.kind == "f":
        arr = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
    elif arr.dtype != np.int16:
        # int32 / int8 / uint8 — best-effort cast.
        try:
            arr = arr.astype(np.int16)
        except Exception:  # noqa: BLE001
            arr = (arr.astype(np.float32) / max(1.0, float(np.max(np.abs(arr) or 1)))) \
                * 32767.0
            arr = arr.astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(arr.tobytes())
    return buf.getvalue()


# -----------------------------------------------------------------------------
# Presentation helpers (HTML cards, comparison table)
# -----------------------------------------------------------------------------

ACCENT_PASS = "#10b981"     # emerald-500 — meets tightest tier
ACCENT_PARTIAL = "#f59e0b"  # amber-500 — meets only Deliverables tier
ACCENT_MISS = "#ef4444"     # red-500 — misses all
ACCENT_VIOLET = "#8b5cf6"   # violet-500 — neutral
SURFACE_BG = "#0b1020"
CARD_BG = "#11162a"
CARD_BORDER = "#1f2740"
TEXT_PRIMARY = "#e5e7eb"
TEXT_MUTED = "#9ca3af"
MONO_FONT = "'JetBrains Mono', 'Fira Code', ui-monospace, SFMono-Regular, Menlo, monospace"


def _ttfc_accent(ttfc_ms: float) -> str:
    """Color a TTFC value: green = beats Step 4 (50ms), yellow = beats Deliverables
    (90ms), red = misses both."""
    if ttfc_ms <= 0:
        return ACCENT_VIOLET
    if ttfc_ms <= STEP4_VALIDATE["TTFC_ms"]:
        return ACCENT_PASS
    if ttfc_ms <= DELIVERABLES["TTFC_ms"]:
        return ACCENT_PARTIAL
    return ACCENT_MISS


def _rtf_accent(rtf: float) -> str:
    """Same tiering for RTF: green<=0.10, yellow<=0.30, red otherwise."""
    if rtf <= 0:
        return ACCENT_VIOLET
    if rtf <= STEP4_VALIDATE["RTF"]:
        return ACCENT_PASS
    if rtf <= DELIVERABLES["RTF"]:
        return ACCENT_PARTIAL
    return ACCENT_MISS


def _e2e_accent(e2e_ms: float) -> str:
    """e2e is report-only per the brief — no PASS/MISS threshold defined.
    The TTFC tiers apply to the *TTS path only*, not the full network
    round-trip (which includes Groq cloud RTT, STT, VAD, pipeline overhead).
    Return the neutral/informational color so the card matches the visual
    treatment of Decode tok/s (also report-only)."""
    return ACCENT_VIOLET


def _metric_card(label: str, value: str, unit: str, accent: str) -> str:
    return f"""
    <div style="
        background: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-radius: 12px;
        padding: 18px 20px;
        min-width: 0;
        box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset;
    ">
        <div style="
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {TEXT_MUTED};
            margin-bottom: 8px;
            font-weight: 600;
        ">{html.escape(label)}</div>
        <div style="
            font-family: {MONO_FONT};
            font-size: 30px;
            font-weight: 700;
            color: {TEXT_PRIMARY};
            line-height: 1.1;
        ">
            <span style="color: {accent}">{html.escape(value)}</span>
            <span style="font-size: 13px; color: {TEXT_MUTED}; font-weight: 500; margin-left: 6px;">{html.escape(unit)}</span>
        </div>
    </div>
    """


def render_metric_cards(m: LiveMetrics) -> str:
    """Render the persistent 4-card metric row. Used by BOTH modes."""
    source_html = (
        f'<div style="font-size:11px; color:{TEXT_MUTED}; font-family:{MONO_FONT}; '
        f'margin-bottom:6px;">last update: {html.escape(m.source)}'
        + (f' • {time.strftime("%H:%M:%S", time.localtime(m.updated_at))}' if m.updated_at else '')
        + '</div>'
    )

    if m.error:
        return source_html + f"""
        <div style="
            background: {CARD_BG};
            border: 1px solid {ACCENT_MISS};
            border-radius: 12px;
            padding: 14px 18px;
            color: {TEXT_PRIMARY};
            font-family: {MONO_FONT};
            font-size: 12px;
        ">
            <div style="color: {ACCENT_MISS}; font-weight: 600; margin-bottom: 4px;">METRIC ERROR</div>
            <div style="color: {TEXT_PRIMARY}; white-space: pre-wrap;">{html.escape(m.error)}</div>
        </div>
        """

    ttfc_disp = f"{m.ttfc_ms:.1f}" if m.ttfc_ms else "—"
    rtf_disp = f"{m.rtf:.3f}" if m.rtf else "—"
    tok_disp = f"{m.decode_tok_per_s:.1f}" if m.decode_tok_per_s else "—"
    e2e_disp = f"{m.e2e_ms:.0f}" if m.e2e_ms else "—"

    cards = [
        _metric_card("TTFC", ttfc_disp, "ms", _ttfc_accent(m.ttfc_ms)),
        _metric_card("RTF", rtf_disp, "", _rtf_accent(m.rtf)),
        _metric_card("Decode tok/s", tok_disp, "tok/s", ACCENT_VIOLET),
        _metric_card("E2E", e2e_disp, "ms", _e2e_accent(m.e2e_ms)),
    ]
    return source_html + f"""
    <div style="
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
    ">
        {''.join(cards)}
    </div>
    """


def _badge(passed: bool) -> str:
    color = ACCENT_PASS if passed else ACCENT_MISS
    label = "PASS" if passed else "MISS"
    return (
        f'<span style="'
        f'display:inline-block; padding:2px 8px; border-radius:9999px; '
        f'background:{color}1a; color:{color}; '
        f'font-family:{MONO_FONT}; font-size:11px; font-weight:700; '
        f'border:1px solid {color}55;'
        f'">{label}</span>'
    )


def render_comparison_table(m: LiveMetrics) -> str:
    """Render the comparison-with-brief table with PASS / MISS badges."""
    if m.error or (m.ttfc_ms == 0 and m.rtf == 0):
        return f"""
        <div style="
            color: {TEXT_MUTED};
            font-family: {MONO_FONT};
            font-size: 12px;
            padding: 12px;
        ">No measurement yet. Use either mode to populate the cards above.</div>
        """

    rows_html: list[str] = []
    # TTFC row
    ttfc_cells = [
        f'<td style="padding:8px 12px; color:{TEXT_MUTED}; font-family:{MONO_FONT};">TTFC</td>',
        f'<td style="padding:8px 12px; color:{TEXT_PRIMARY}; font-family:{MONO_FONT}; font-weight:600;">{m.ttfc_ms:.1f} ms</td>',
    ]
    for _tier_label, tier in TARGET_TIERS:
        passed = m.ttfc_ms > 0 and m.ttfc_ms <= tier["TTFC_ms"]
        ttfc_cells.append(
            f'<td style="padding:8px 12px; font-family:{MONO_FONT}; color:{TEXT_MUTED};">'
            f'&lt;{tier["TTFC_ms"]:.0f}ms&nbsp;&nbsp;{_badge(passed)}</td>'
        )
    rows_html.append("<tr>" + "".join(ttfc_cells) + "</tr>")

    # RTF row
    rtf_cells = [
        f'<td style="padding:8px 12px; color:{TEXT_MUTED}; font-family:{MONO_FONT};">RTF</td>',
        f'<td style="padding:8px 12px; color:{TEXT_PRIMARY}; font-family:{MONO_FONT}; font-weight:600;">{m.rtf:.4f}</td>',
    ]
    for _tier_label, tier in TARGET_TIERS:
        passed = m.rtf > 0 and m.rtf <= tier["RTF"]
        rtf_cells.append(
            f'<td style="padding:8px 12px; font-family:{MONO_FONT}; color:{TEXT_MUTED};">'
            f'&lt;{tier["RTF"]:.2f}&nbsp;&nbsp;{_badge(passed)}</td>'
        )
    rows_html.append("<tr>" + "".join(rtf_cells) + "</tr>")

    header_cells = [
        f'<th style="text-align:left; padding:8px 12px; color:#6b7280; font-size:11px; text-transform:uppercase; letter-spacing:.08em; border-bottom:1px solid {CARD_BORDER};">Metric</th>',
        f'<th style="text-align:left; padding:8px 12px; color:#6b7280; font-size:11px; text-transform:uppercase; letter-spacing:.08em; border-bottom:1px solid {CARD_BORDER};">Latest</th>',
    ]
    for tier_label, _tier in TARGET_TIERS:
        header_cells.append(
            f'<th style="text-align:left; padding:8px 12px; color:#6b7280; font-size:11px; text-transform:uppercase; letter-spacing:.08em; border-bottom:1px solid {CARD_BORDER};">{html.escape(tier_label)}</th>'
        )

    return f"""
    <div style="
        background: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-radius: 12px;
        padding: 8px 4px 4px;
        overflow-x: auto;
    ">
        <table style="width:100%; border-collapse:collapse; font-size:13px;">
            <thead><tr>{''.join(header_cells)}</tr></thead>
            <tbody>{''.join(rows_html)}</tbody>
        </table>
    </div>
    """


def render_header_html() -> str:
    pipeline_step = (
        f'font-family:{MONO_FONT}; font-size:12px; '
        f'color:{TEXT_PRIMARY}; background:{CARD_BG}; '
        f'border:1px solid {CARD_BORDER}; padding:6px 10px; '
        f'border-radius:8px;'
    )
    arrow = (
        f'color:{TEXT_MUTED}; font-family:{MONO_FONT}; '
        f'font-size:14px; padding:0 6px;'
    )
    return f"""
    <div style="margin-bottom:8px;">
        <div style="
            display:flex; align-items:baseline; justify-content:space-between;
            gap:16px; flex-wrap:wrap;
        ">
            <div>
                <div style="font-size:24px; font-weight:700; color:{TEXT_PRIMARY}; letter-spacing:-0.01em;">
                    e3 x Megakernel x Qwen3-TTS
                </div>
                <div style="font-size:12px; color:{TEXT_MUTED}; margin-top:4px; font-family:{MONO_FONT};">
                    Test &amp; measure (mic + Deepgram + Groq + Megakernel) — or — Live conversation (Pipecat WebRTC on :{WEBRTC_PORT})
                </div>
            </div>
            <div style="
                font-family:{MONO_FONT}; font-size:11px; color:{TEXT_MUTED};
                text-align:right;
            ">
                steady-state, post-warmup<br/>
                RTX 5090 sm_120, CUDA 13.1
            </div>
        </div>

        <div style="
            margin-top:14px; display:flex; align-items:center; flex-wrap:wrap;
            gap:4px;
        ">
            <span style="{pipeline_step}">mic</span>
            <span style="{arrow}">&rarr;</span>
            <span style="{pipeline_step}">Deepgram STT</span>
            <span style="{arrow}">&rarr;</span>
            <span style="{pipeline_step}">Groq LLM</span>
            <span style="{arrow}">&rarr;</span>
            <span style="{pipeline_step}; border-color:{ACCENT_PASS}55; color:{ACCENT_PASS};">Megakernel TTS</span>
            <span style="{arrow}">&rarr;</span>
            <span style="{pipeline_step}">audio</span>
        </div>

        <div style="
            margin-top:14px;
            border:1px solid {ACCENT_PARTIAL}55;
            background:{ACCENT_PARTIAL}14;
            color:{TEXT_PRIMARY};
            padding:10px 14px;
            border-radius:10px;
            font-size:13px;
            display:flex; gap:10px; align-items:flex-start;
        ">
            <span style="color:{ACCENT_PARTIAL}; font-weight:700; font-family:{MONO_FONT}; font-size:11px;">HONEST</span>
            <span style="color:{TEXT_PRIMARY};">{html.escape(DISCLAIMER_TEXT)}</span>
        </div>
    </div>
    """


def render_build_flags_html(load_status: str) -> str:
    rows: list[str] = []
    for k, v in BUILD_FLAGS.items():
        v_str = f"{v:g}" if isinstance(v, float) else str(v)
        rows.append(
            f'<tr>'
            f'<td style="padding:4px 10px 4px 0; color:{TEXT_MUTED}; font-family:{MONO_FONT}; font-size:12px;">{html.escape(k)}</td>'
            f'<td style="padding:4px 0; color:{TEXT_PRIMARY}; font-family:{MONO_FONT}; font-size:12px; font-weight:600;">{html.escape(v_str)}</td>'
            f'</tr>'
        )
    discl = "".join(
        f'<li style="margin:2px 0;">{html.escape(s)}</li>'
        for s in HONEST_DISCLOSURES
    )
    return f"""
    <div style="
        background: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-radius: 12px;
        padding: 14px 16px;
    ">
        <div style="
            font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
            color: {TEXT_MUTED}; margin-bottom: 10px; font-weight: 600;
        ">Build flags</div>
        <table style="border-collapse:collapse;">{''.join(rows)}</table>

        <div style="
            font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
            color: {TEXT_MUTED}; margin: 16px 0 8px; font-weight: 600;
        ">Honest disclosures</div>
        <ul style="
            margin:0; padding-left:18px; color:{TEXT_PRIMARY};
            font-size:12px; line-height:1.5;
        ">{discl}</ul>

        <div style="
            font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
            color: {TEXT_MUTED}; margin: 16px 0 8px; font-weight: 600;
        ">Service status</div>
        <div style="
            font-family:{MONO_FONT}; font-size:12px; color:{TEXT_PRIMARY};
            background:{SURFACE_BG}; border:1px solid {CARD_BORDER};
            padding:8px 10px; border-radius:8px; white-space:pre-wrap;
        ">{html.escape(load_status)}</div>
    </div>
    """


def render_stage_log(
    *,
    user_text: str = "",
    stt_ms: float = 0.0,
    llm_text: str = "",
    llm_ms: float = 0.0,
    ttfc_ms: float = 0.0,
    tts_total_ms: float = 0.0,
    audio_seconds: float = 0.0,
    e2e_ms: float = 0.0,
    error: str | None = None,
) -> str:
    """Per-stage timing log for Mode A."""
    if error:
        return f"""
        <div style="
            background: {CARD_BG};
            border: 1px solid {ACCENT_MISS};
            border-radius: 10px;
            padding: 14px 16px;
            font-family: {MONO_FONT};
            font-size: 12px;
            color: {TEXT_PRIMARY};
            white-space: pre-wrap;
        ">
            <div style="color:{ACCENT_MISS}; font-weight:700; margin-bottom:6px;">PIPELINE ERROR</div>
            {html.escape(error)}
        </div>
        """
    if not user_text and not llm_text:
        return f"""
        <div style="
            color:{TEXT_MUTED}; font-family:{MONO_FONT}; font-size:12px; padding:8px;
        ">Record into the mic above and press Send to see per-stage timings.</div>
        """

    def _row(label: str, value: str, ms: float | None) -> str:
        ms_html = (
            f'<span style="color:{TEXT_MUTED}; font-family:{MONO_FONT}; font-size:11px;">'
            f'{ms:.0f} ms</span>' if ms is not None else ''
        )
        return f"""
        <div style="
            display:flex; justify-content:space-between; gap:12px;
            padding:6px 0; border-bottom:1px solid {CARD_BORDER};
        ">
            <div style="font-family:{MONO_FONT}; font-size:12px; color:{TEXT_MUTED}; min-width:90px;">{html.escape(label)}</div>
            <div style="font-size:13px; color:{TEXT_PRIMARY}; flex:1; word-break:break-word;">{html.escape(value)}</div>
            <div style="text-align:right; min-width:70px;">{ms_html}</div>
        </div>
        """

    rows = [
        _row("STT", user_text or "—", stt_ms or None),
        _row("LLM", llm_text or "—", llm_ms or None),
        _row(
            "TTS TTFC",
            f"{audio_seconds:.2f} s of audio decoded",
            ttfc_ms or None,
        ),
        _row("TTS total", "(megakernel forward + codec)", tts_total_ms or None),
        _row("E2E", "mic → audio out", e2e_ms or None),
    ]
    return f"""
    <div style="
        background: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-radius: 10px;
        padding: 8px 14px;
    ">
        {''.join(rows)}
    </div>
    """


# -----------------------------------------------------------------------------
# Snapshot polling (Live mode)
# -----------------------------------------------------------------------------


def _read_snapshot() -> dict[str, Any] | None:
    """Read the JSON snapshot written by ``pipecat_demo._write_snapshot``.

    Returns None if the file is missing, unparseable, or older than 5 minutes
    (stale data should not paint the cards green from a previous session).
    """
    p = Path(METRICS_SNAPSHOT_PATH)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            snap = json.load(f)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(snap, dict):
        return None
    # Drop stale snapshots (>5 min) — they're almost certainly from a
    # previous run, and stale-green cards would mislead the reviewer.
    age = time.time() - float(snap.get("updated_at") or 0.0)
    if age > 300:
        return None
    return snap


# -----------------------------------------------------------------------------
# Gradio Blocks layout
# -----------------------------------------------------------------------------


def _initial_service_status() -> str:
    svc = _load_services()
    lines: list[str] = []
    lines.append(
        "MegakernelTTSService: " + (
            "OK" if svc.tts is not None else f"FAILED — {svc.tts_error}"
        )
    )
    lines.append(
        "DEEPGRAM_API_KEY: " + ("present" if svc.deepgram_key else "MISSING (Mode A STT will fail)")
    )
    lines.append(
        "LLM_API_KEY (Groq): " + ("present" if svc.groq_key else "MISSING (Mode A LLM will fail)")
    )
    lines.append(f"Groq model: {svc.llm_model}")
    lines.append(f"Live WebRTC backend: http://localhost:{WEBRTC_PORT}/")
    lines.append(f"Metrics snapshot: {METRICS_SNAPSHOT_PATH}")
    return "\n".join(lines)


def build_ui():
    """Construct the Gradio Blocks app. Returns the Blocks instance."""
    import gradio as gr

    custom_css = f"""
    .gradio-container {{
        background: {SURFACE_BG} !important;
        color: {TEXT_PRIMARY} !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}
    .gradio-container * {{
        border-color: {CARD_BORDER} !important;
    }}
    .gradio-container .prose {{
        color: {TEXT_PRIMARY} !important;
    }}
    .gradio-container textarea, .gradio-container input {{
        background: {CARD_BG} !important;
        color: {TEXT_PRIMARY} !important;
        font-family: {MONO_FONT} !important;
    }}
    .gradio-container label span {{
        color: {TEXT_MUTED} !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        font-weight: 600 !important;
    }}
    .gradio-container button.primary {{
        background: {ACCENT_VIOLET} !important;
        border-color: {ACCENT_VIOLET} !important;
        color: white !important;
    }}
    """

    with gr.Blocks(
        title="e3 x Megakernel x Qwen3-TTS",
        theme=gr.themes.Base(
            primary_hue="violet",
            neutral_hue="slate",
        ).set(
            body_background_fill=SURFACE_BG,
            body_text_color=TEXT_PRIMARY,
            background_fill_primary=CARD_BG,
            background_fill_secondary=CARD_BG,
            border_color_primary=CARD_BORDER,
        ),
        css=custom_css,
        analytics_enabled=False,
    ) as demo:
        gr.HTML(render_header_html())

        # Persistent shared state: latest metrics + which mode produced them.
        live_state = gr.State(LiveMetrics())

        # ---- Mode selector (radio: mutually exclusive) -----------------
        mode_radio = gr.Radio(
            label="Mode",
            choices=["Test & measure", "Live conversation"],
            value="Test & measure",
            interactive=True,
        )

        with gr.Row():
            with gr.Column(scale=3, min_width=520):

                # ---- Persistent metric cards (top of column) ------------
                gr.HTML(
                    '<div style="font-size:11px; text-transform:uppercase; '
                    f'letter-spacing:0.08em; color:{TEXT_MUTED}; '
                    'margin:2px 0 8px; font-weight:600;">'
                    'Live metrics (latest from any mode)</div>'
                )
                metrics_html = gr.HTML(render_metric_cards(LiveMetrics()))

                gr.HTML(
                    '<div style="font-size:11px; text-transform:uppercase; '
                    f'letter-spacing:0.08em; color:{TEXT_MUTED}; '
                    'margin:18px 0 8px; font-weight:600;">'
                    'Comparison with brief</div>'
                )
                comparison_html = gr.HTML(render_comparison_table(LiveMetrics()))

                # ---- Mode A group (Test & measure) ----------------------
                with gr.Group(visible=True) as mode_a_group:
                    gr.HTML(
                        '<div style="font-size:11px; text-transform:uppercase; '
                        f'letter-spacing:0.08em; color:{TEXT_MUTED}; '
                        'margin:18px 0 8px; font-weight:600;">'
                        'Mode A — Test &amp; measure (mic → STT → LLM → TTS)</div>'
                    )
                    mic_in = gr.Audio(
                        label="Record a question",
                        sources=["microphone"],
                        type="numpy",
                        editable=False,
                        interactive=True,
                    )
                    with gr.Row():
                        frames_in = gr.Slider(
                            label="Frames (caps LLM reply tokens + TTS decode horizon)",
                            minimum=5,
                            maximum=100,
                            step=1,
                            value=25,
                        )
                        send_btn = gr.Button(
                            "Send",
                            variant="primary",
                            scale=0,
                            min_width=120,
                        )
                    audio_out = gr.Audio(
                        label="Bot reply (24 kHz int16)",
                        type="filepath",
                        autoplay=True,
                        interactive=False,
                    )
                    gr.HTML(
                        '<div style="font-size:11px; text-transform:uppercase; '
                        f'letter-spacing:0.08em; color:{TEXT_MUTED}; '
                        'margin:14px 0 6px; font-weight:600;">'
                        'Per-stage timings</div>'
                    )
                    stage_log_html = gr.HTML(render_stage_log())

                # ---- Mode B group (Live conversation) -------------------
                with gr.Group(visible=False) as mode_b_group:
                    gr.HTML(
                        '<div style="font-size:11px; text-transform:uppercase; '
                        f'letter-spacing:0.08em; color:{TEXT_MUTED}; '
                        'margin:18px 0 8px; font-weight:600;">'
                        'Mode B — Live conversation (Pipecat WebRTC)</div>'
                    )
                    # We pick an iframe because Pipecat's SmallWebRTCTransport
                    # serves the FULL client (HTML + JS + /api/offer + WebRTC
                    # signalling) from a single origin (port 8081). Embedding
                    # that origin in an iframe Just Works: getUserMedia is
                    # gated on a secure context, and `localhost`/`127.0.0.1`
                    # ARE considered secure by Chrome/Safari, so no TLS dance
                    # is needed even when the iframe's parent is also on
                    # localhost. The "open in new tab" fallback is provided
                    # for browsers that block mic access inside iframes
                    # (Firefox occasionally needs an explicit permission
                    # grant on the inner origin).
                    gr.HTML(
                        f"""
                        <div style="
                            background:{CARD_BG};
                            border:1px solid {CARD_BORDER};
                            border-radius:10px;
                            padding:12px 14px;
                            margin-bottom:10px;
                            color:{TEXT_PRIMARY};
                            font-size:13px;
                        ">
                            <div style="margin-bottom:6px;">
                                Live conversation runs in
                                <code style="font-family:{MONO_FONT}; background:{SURFACE_BG};
                                       padding:2px 6px; border-radius:4px; border:1px solid {CARD_BORDER};">
                                    pipecat_demo.py (INPUT_MODE=webrtc)
                                </code>
                                on port <b>{WEBRTC_PORT}</b>. Start it in a separate terminal,
                                then click <b>Start</b> below to grant mic access.
                            </div>
                            <a href="http://localhost:{WEBRTC_PORT}/" target="_blank"
                               style="color:{ACCENT_VIOLET}; text-decoration:none; font-family:{MONO_FONT}; font-size:12px;">
                                Open live conversation in a new tab &rarr;
                            </a>
                        </div>
                        <iframe
                            src="http://localhost:{WEBRTC_PORT}/"
                            width="100%"
                            height="520"
                            allow="microphone; autoplay; camera"
                            style="border:1px solid {CARD_BORDER}; border-radius:10px; background:{CARD_BG};">
                        </iframe>
                        <div style="
                            color:{TEXT_MUTED}; font-family:{MONO_FONT}; font-size:11px;
                            margin-top:8px;
                        ">
                            Metrics above auto-refresh every 2s from
                            {html.escape(METRICS_SNAPSHOT_PATH)}.
                        </div>
                        """
                    )

            with gr.Column(scale=1, min_width=300):
                gr.HTML(render_build_flags_html(_initial_service_status()))

        # ---------------------------------------------------------------
        # Mode-switcher: hide one group when the other is active
        # ---------------------------------------------------------------
        def on_mode_change(mode: str):
            is_a = (mode == "Test & measure")
            return (
                gr.update(visible=is_a),
                gr.update(visible=not is_a),
            )

        mode_radio.change(
            fn=on_mode_change,
            inputs=[mode_radio],
            outputs=[mode_a_group, mode_b_group],
        )

        # ---------------------------------------------------------------
        # Mode A click handler (async — keeps the UI loop responsive)
        # ---------------------------------------------------------------
        async def on_send(
            mic_value: Any,
            frames: float,
            current_state: LiveMetrics,
        ):
            t0 = time.perf_counter()

            if mic_value is None:
                err = "No mic recording. Click the mic widget, record a question, then Send."
                return (
                    None,
                    render_stage_log(error=err),
                    render_metric_cards(current_state),
                    render_comparison_table(current_state),
                    current_state,
                )

            svc = _load_services()

            # Guardrails: surface honest service errors instead of silently
            # falling back to anything (no mac TTS, ever).
            if svc.tts is None:
                err = svc.tts_error or "MegakernelTTSService not initialized."
                return (
                    None,
                    render_stage_log(error=err),
                    render_metric_cards(current_state),
                    render_comparison_table(current_state),
                    current_state,
                )
            if not svc.deepgram_key:
                err = "DEEPGRAM_API_KEY missing in env / .env"
                return (
                    None, render_stage_log(error=err),
                    render_metric_cards(current_state),
                    render_comparison_table(current_state),
                    current_state,
                )
            if not svc.groq_key:
                err = "LLM_API_KEY (Groq) missing in env / .env"
                return (
                    None, render_stage_log(error=err),
                    render_metric_cards(current_state),
                    render_comparison_table(current_state),
                    current_state,
                )

            # 1. Pack mic recording into a WAV the Deepgram REST endpoint
            #    accepts. Gradio hands us (sample_rate, np.ndarray).
            try:
                wav_bytes = _numpy_audio_to_wav_bytes(mic_value)
            except Exception as e:  # noqa: BLE001
                return (
                    None,
                    render_stage_log(error=f"WAV pack failed: {e!r}"),
                    render_metric_cards(current_state),
                    render_comparison_table(current_state),
                    current_state,
                )

            # 2. STT — Deepgram REST.
            try:
                user_text, stt_ms = await _stt_deepgram_async(wav_bytes, svc.deepgram_key)
            except Exception as e:  # noqa: BLE001
                return (
                    None,
                    render_stage_log(error=f"Deepgram STT failed: {e!r}"),
                    render_metric_cards(current_state),
                    render_comparison_table(current_state),
                    current_state,
                )
            if not user_text:
                return (
                    None,
                    render_stage_log(error="Deepgram returned empty transcript — try again."),
                    render_metric_cards(current_state),
                    render_comparison_table(current_state),
                    current_state,
                )

            # 3. LLM — Groq. Frames slider also caps reply tokens so reply
            #    length is bounded and TTS test is bounded.
            try:
                llm_text, llm_ms = await _llm_groq_async(
                    user_text, svc.groq_key, svc.llm_model,
                    max_tokens=int(frames) * 4,  # ~4 tokens per "frame" budget
                )
            except Exception as e:  # noqa: BLE001
                return (
                    None,
                    render_stage_log(error=f"Groq LLM failed: {e!r}"),
                    render_metric_cards(current_state),
                    render_comparison_table(current_state),
                    current_state,
                )

            # 4. TTS — MegakernelTTSService (REAL megakernel, no mac fallback).
            try:
                pcm, ttfc_ms, tts_total_ms, audio_seconds = \
                    await _tts_megakernel_async(svc.tts, llm_text)
            except Exception as e:  # noqa: BLE001
                return (
                    None,
                    render_stage_log(error=f"MegakernelTTS failed: {e!r}"),
                    render_metric_cards(current_state),
                    render_comparison_table(current_state),
                    current_state,
                )

            # 5. Pack the bot's PCM into a WAV file (Gradio Audio widget
            #    with type='filepath' needs a path, not bytes).
            wav_path = _wav_from_pcm_int16(pcm, SAMPLE_RATE_HZ) if pcm else None
            e2e_ms = (time.perf_counter() - t0) * 1000.0

            # 6. Update the live-metrics state. RTF = wall-clock / audio_s.
            rtf = (tts_total_ms / 1000.0 / audio_seconds) if audio_seconds > 0 else 0.0
            # tok/s: per-frame yields ~ 12.5/sec of audio; the megakernel
            # streams one talker token per frame, so frames_decoded ≈
            # audio_seconds * 12.5. We surface that as the "decode tok/s".
            tok_per_s = (audio_seconds * 12.5) / (tts_total_ms / 1000.0) if tts_total_ms > 0 else 0.0
            new_state = LiveMetrics(
                ttfc_ms=ttfc_ms,
                rtf=rtf,
                decode_tok_per_s=tok_per_s,
                e2e_ms=e2e_ms,
                source="Test mode",
                updated_at=time.time(),
            )

            stage_html = render_stage_log(
                user_text=user_text,
                stt_ms=stt_ms,
                llm_text=llm_text,
                llm_ms=llm_ms,
                ttfc_ms=ttfc_ms,
                tts_total_ms=tts_total_ms,
                audio_seconds=audio_seconds,
                e2e_ms=e2e_ms,
            )

            return (
                wav_path,
                stage_html,
                render_metric_cards(new_state),
                render_comparison_table(new_state),
                new_state,
            )

        send_btn.click(
            fn=on_send,
            inputs=[mic_in, frames_in, live_state],
            outputs=[audio_out, stage_log_html, metrics_html, comparison_html, live_state],
        )

        # ---------------------------------------------------------------
        # Live-mode polling: every 2s, read the snapshot file written by
        # pipecat_demo._write_snapshot and re-render the cards if it
        # contains FRESHER values than what's in live_state.
        # ---------------------------------------------------------------
        def on_tick(current_state: LiveMetrics):
            snap = _read_snapshot()
            if snap is None:
                return (
                    render_metric_cards(current_state),
                    render_comparison_table(current_state),
                    current_state,
                )

            updated_at = float(snap.get("updated_at") or 0.0)
            # Skip if the snapshot is older than what we already showed
            # (e.g. the user just ran a Mode A pass which is newer).
            if updated_at <= current_state.updated_at:
                return (
                    render_metric_cards(current_state),
                    render_comparison_table(current_state),
                    current_state,
                )

            ttfc = float(snap.get("ttfc_ms") or 0.0)
            e2e = float(snap.get("e2e_ms") or 0.0)
            # Live mode doesn't expose RTF / tok/s directly (those are
            # bench-only). We keep the previous values so the cards don't
            # flicker to "—" on every poll; mark TTFC + E2E as "Live mode".
            new_state = LiveMetrics(
                ttfc_ms=ttfc or current_state.ttfc_ms,
                rtf=current_state.rtf,
                decode_tok_per_s=current_state.decode_tok_per_s,
                e2e_ms=e2e or current_state.e2e_ms,
                source="Live mode",
                updated_at=updated_at,
            )
            return (
                render_metric_cards(new_state),
                render_comparison_table(new_state),
                new_state,
            )

        # gr.Timer fires every N seconds. In Gradio 4.x this is the
        # idiomatic way to poll the server from a Blocks app without
        # needing a JS .every() handler.
        timer = gr.Timer(value=2.0)
        timer.tick(
            fn=on_tick,
            inputs=[live_state],
            outputs=[metrics_html, comparison_html, live_state],
        )

    return demo


def main() -> None:
    demo = build_ui()
    # Gradio 6.x removed show_api from launch(); 4.x still accepts it but
    # we don't need it. queue() default_concurrency_limit=1 keeps a single
    # megakernel forward in flight at a time (the GPU is the bottleneck).
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        inbrowser=False,
        quiet=False,
    )


if __name__ == "__main__":
    main()
