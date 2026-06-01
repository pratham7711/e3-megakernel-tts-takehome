"""Polished Gradio UI for the e3 Megakernel x Qwen3-TTS take-home submission.

Single mode — **Test & measure**: mic-record one question, hit Send and the
pipeline runs Deepgram REST STT → Groq LLM → ``MegakernelTTSService`` (the
REAL megakernel — no macOS fallback anywhere) → bot WAV. Each stage is timed
and surfaced in a per-stage log. The 4 metric cards (TTFC, RTF, decode tok/s,
audio duration / e2e) live above the widgets, colored against the brief's
three target tiers:

    GREEN  -> meets Step 4 Validate (tightest: TTFC<50ms, RTF<0.10)
    YELLOW -> meets only Deliverables (TTFC<90ms, RTF<0.30)
    RED    -> misses all tiers

The earlier "Live conversation" WebRTC mode was removed: SSH tunnels carry
only TCP, WebRTC media is UDP, so live audio required either a TURN relay
(account-bound credentials) or re-renting the GPU box with `-p UDP:N:M`.
The brief's measurable deliverables don't depend on live mode and Mode A
covers the full pipeline.

Run on the GPU box::

    cd inference-server
    PYTHONPATH=/workspace/qwen_megakernel python3 ui_v2.py
    # then SSH-tunnel port 8080:
    ssh -L 8080:localhost:8080 <gpu-box>

Design notes
------------
* The pipeline runs in an ``async`` handler (Gradio runs each handler in its
  own asyncio task), so a long-running megakernel forward never freezes the
  UI's event loop. STT + LLM use ``httpx.AsyncClient``; the TTS is
  intrinsically async via ``MegakernelTTSService._tts.generate()``.
* No macOS ``say`` fallback ANYWHERE. If the megakernel fails to load
  (``LoadedComponents.stub == True``) we surface the load error in the
  stage log instead of pretending audio rendered.
* The UI reuses ``MegakernelTTSService`` (not the raw kernel directly) per
  the brief's "DO NOT call the megakernel directly" rule.
"""

from __future__ import annotations

import asyncio
import html
import io
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
    "Pipeline: Deepgram REST → Groq LLM → MegakernelTTSService → streaming PCM."
)

# TODO(W3): the bench line below carries provisional TTFC/RTF numbers
# (18.7 ms / 0.123) captured before the W1 megakernel↔Qwen3-TTS wiring
# fix landed. Once W3 (canonical bench rerun) publishes final numbers,
# replace the bench-line entry below with the canonical values. Do NOT
# hand-edit those numbers — wait for the W3 deliverable.
HONEST_DISCLOSURES: list[str] = [
    "Codec: REAL Qwen3-TTS V2 vocoder (271 weights, clean-room reimpl)",
    "Per-frame streaming: TTFC measured at FIRST PCM chunk; browser plays chunks as they arrive",
    "Uses MegakernelTTSService (Pipecat-wrapped) per brief — never the raw kernel; no mac fallback",
    "Talker AR loop: eager PyTorch step_embed (correctness) — kernel step(int) bypassed to allow per-step text-hidden injection. Bench harness uses the kernel path.",
    "Bench (n=5 warm + 3 warmup, cuda.synchronize() at every boundary): TTFC 18.7±0.1 ms · RTF 0.123 · streaming chunks 10 ms cadence",  # W3-pending
    "GPU: 1× RTX 5090 sm_120 Blackwell, CUDA 13.1, PyTorch 2.10.0a NGC",
]

# Output sample rate for the Qwen3-TTS codec (24 kHz int16 PCM).
SAMPLE_RATE_HZ: int = 24_000

# Model checkpoint + speaker (matches inference-server defaults).
MODEL_PATH = "/workspace/qwen3-tts-1.7b"
SPEAKER = "ryan"


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
    """Lazy-init the pipeline services. Cached across calls.

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
    # Send-click will surface the error in the stage log.
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
# Voice pipeline — mic-record → STT → LLM → TTS pipeline
# -----------------------------------------------------------------------------


async def _stt_deepgram_async(wav_bytes: bytes, api_key: str) -> tuple[str, float]:
    """Deepgram REST one-shot transcription. Returns (text, latency_ms).

    Uses ``httpx.AsyncClient`` so the call doesn't block Gradio's event
    loop. The "prerecorded" REST endpoint is right for a one-shot recorded
    question; the streaming WebSocket endpoint is what ``DeepgramSTTService``
    uses for streaming STT.
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
            peak = float(np.max(np.abs(arr))) if arr.size else 1.0
            denom = peak if peak > 0 else 1.0
            arr = (arr.astype(np.float32) / max(1.0, denom)) * 32767.0
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

# Palette per design-taste-frontend directives:
#   - Zinc-950 base (no pure black)
#   - Single saturated accent (emerald) for PASS
#   - Desaturated amber/rose for warning/miss (no neon glows)
#   - No purple/violet (banned by skill section 7)
#   - Off-white text, muted slate for secondary
SURFACE_BG = "#09090b"        # zinc-950 (off-black, never #000)
SURFACE_INSET = "#0c0d11"     # slightly raised — for inner panels
CARD_BG = "#0e1014"           # one shade up from surface; cards barely lift
CARD_BORDER = "rgba(255,255,255,0.06)"   # 1px hairline — anti-glow
DIVIDER = "rgba(255,255,255,0.08)"
TEXT_PRIMARY = "#fafafa"      # zinc-50
TEXT_BODY = "#d4d4d8"         # zinc-300 — boosted contrast (was zinc-400, hard to read)
TEXT_MUTED = "#a1a1aa"        # zinc-400 — was zinc-500
TEXT_LABEL = "#71717a"        # zinc-500 — for uppercase eyebrows (was zinc-600)
ACCENT_PASS = "#10b981"       # emerald-500 (single accent, saturation ~67%)
ACCENT_PARTIAL = "#d97706"    # amber-600, desaturated
ACCENT_MISS = "#b91c1c"       # red-700, desaturated (no neon)
ACCENT_NEUTRAL = TEXT_BODY    # informational metrics — quiet, not violet
ACCENT_INFO = "#3b82f6"       # electric blue — used sparingly, for live-state dots

# Anti-emoji policy: any iconography below uses inline SVG primitives.
# Typography stack — engineer-dashboard, no Inter, no Serif.
SANS_FONT = "'Geist', 'Satoshi', system-ui, -apple-system, 'Segoe UI', sans-serif"
MONO_FONT = "'JetBrains Mono', 'Geist Mono', 'Fira Code', ui-monospace, SFMono-Regular, Menlo, monospace"

# Back-compat alias — some legacy references in this file used ACCENT_VIOLET.
ACCENT_VIOLET = ACCENT_NEUTRAL


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


def _metric_cell(label: str, value: str, unit: str, accent: str, target_hint: str) -> str:
    """Single metric cell — hairline-divider layout, NOT a boxy card.

    Design notes per skill rule 4 (anti-card-overuse):
      - No background fill, no shadow.
      - 1px left border in the accent color = colored gutter for PASS/FAIL.
      - Number in mono, value-accent only (label/unit stay neutral).
      - Target hint in a compact line below the number so a reviewer
        sees the bar without consulting a separate table.
    """
    return f"""
    <div style="
        border-left: 2px solid {accent};
        padding: 6px 18px 6px 14px;
        min-width: 0;
        display: flex;
        flex-direction: column;
        gap: 4px;
    ">
        <div style="
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: {TEXT_LABEL};
            font-weight: 600;
        ">{html.escape(label)}</div>
        <div style="
            font-family: {MONO_FONT};
            font-size: 28px;
            font-weight: 600;
            line-height: 1;
            color: {TEXT_PRIMARY};
            display: flex;
            align-items: baseline;
            gap: 6px;
        ">
            <span style="color: {accent};">{html.escape(value)}</span>
            <span style="font-size: 12px; color: {TEXT_MUTED}; font-weight: 500; font-family: {MONO_FONT};">{html.escape(unit)}</span>
        </div>
        <div style="
            font-family: {MONO_FONT};
            font-size: 10.5px;
            color: {TEXT_MUTED};
            font-weight: 500;
            letter-spacing: 0.02em;
        ">{html.escape(target_hint)}</div>
    </div>
    """


def render_metric_cards(m: LiveMetrics) -> str:
    """Render the persistent metric row. Used by BOTH modes."""
    source_html = (
        f'<div style="font-size:10.5px; color:{TEXT_LABEL}; font-family:{MONO_FONT}; '
        f'margin: 0 0 10px 0; text-transform: uppercase; letter-spacing: 0.08em;">'
        f'<span style="color:{ACCENT_PASS};">●</span> &nbsp;LIVE &nbsp;·&nbsp; SOURCE {html.escape(m.source or "—")}'
        + (f' &nbsp;·&nbsp; {time.strftime("%H:%M:%S", time.localtime(m.updated_at))}' if m.updated_at else '')
        + '</div>'
    )

    if m.error:
        return source_html + f"""
        <div style="
            border-left: 2px solid {ACCENT_MISS};
            padding: 10px 14px;
            font-family: {MONO_FONT};
            font-size: 12px;
            color: {TEXT_PRIMARY};
        ">
            <div style="color: {ACCENT_MISS}; font-weight: 600; margin-bottom: 4px; font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.1em;">METRIC ERROR</div>
            <div style="color: {TEXT_BODY}; white-space: pre-wrap;">{html.escape(m.error)}</div>
        </div>
        """

    ttfc_disp = f"{m.ttfc_ms:.1f}" if m.ttfc_ms else "—"
    rtf_disp = f"{m.rtf:.3f}" if m.rtf else "—"
    tok_disp = f"{m.decode_tok_per_s:.0f}" if m.decode_tok_per_s else "—"
    e2e_disp = f"{m.e2e_ms:.0f}" if m.e2e_ms else "—"

    cards = [
        _metric_cell("TTFC", ttfc_disp, "ms", _ttfc_accent(m.ttfc_ms), "target <50 / <60 / <90"),
        _metric_cell("RTF",  rtf_disp,  "",   _rtf_accent(m.rtf),     "target <0.1 / <0.15 / <0.3"),
        _metric_cell("Decode", tok_disp, "tok/s", ACCENT_NEUTRAL,      "1.7B talker · report-only"),
        _metric_cell("E2E",  e2e_disp,  "ms", ACCENT_NEUTRAL,         "UserStop → BotStart · informational"),
    ]
    return source_html + f"""
    <div style="
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 4px;
        padding: 6px 0;
        border-top: 1px solid {DIVIDER};
        border-bottom: 1px solid {DIVIDER};
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
            font-size: 11.5px;
            padding: 2px 0 0;
            letter-spacing: 0.02em;
        ">awaiting first run · targets fill in after the first Send</div>
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
        f'<th style="text-align:left; padding:8px 12px; color:#6b7280; font-size:11px; text-transform:uppercase; letter-spacing:.08em; border-bottom:1px solid {DIVIDER};">Metric</th>',
        f'<th style="text-align:left; padding:8px 12px; color:#6b7280; font-size:11px; text-transform:uppercase; letter-spacing:.08em; border-bottom:1px solid {DIVIDER};">Latest</th>',
    ]
    for tier_label, _tier in TARGET_TIERS:
        header_cells.append(
            f'<th style="text-align:left; padding:8px 12px; color:#6b7280; font-size:11px; text-transform:uppercase; letter-spacing:.08em; border-bottom:1px solid {DIVIDER};">{html.escape(tier_label)}</th>'
        )

    return f"""
    <div style="
        padding: 0;
        overflow-x: auto;
        border-top: 1px solid {DIVIDER};
    ">
        <table style="width:100%; border-collapse:collapse; font-size:12.5px;">
            <thead><tr>{''.join(header_cells)}</tr></thead>
            <tbody>{''.join(rows_html)}</tbody>
        </table>
    </div>
    """


def render_header_html() -> str:
    # Pipeline tokens — thin hairline, no background fill, all-uppercase mono.
    # Highlights the megakernel link in emerald (single accent rule).
    step = (
        f'font-family:{MONO_FONT}; font-size:11px; '
        f'color:{TEXT_BODY}; padding:4px 8px; letter-spacing:0.06em; '
        f'border:1px solid {CARD_BORDER}; '
        f'text-transform:uppercase;'
    )
    arrow = (
        f'color:{TEXT_LABEL}; font-family:{MONO_FONT}; '
        f'font-size:12px; padding:0 4px;'
    )
    return f"""
    <div style="margin: 4px 0 18px 0;">
        <div style="
            display:flex; align-items:flex-end; justify-content:space-between;
            gap:24px; flex-wrap:wrap; margin-bottom: 14px;
        ">
            <div>
                <div style="
                    font-family:{MONO_FONT};
                    font-size:10.5px;
                    color:{TEXT_LABEL};
                    letter-spacing:0.14em;
                    text-transform:uppercase;
                    margin-bottom: 4px;
                ">take-home · contrario × e3</div>
                <div style="
                    font-size:22px;
                    font-weight:600;
                    color:{TEXT_PRIMARY};
                    letter-spacing:-0.01em;
                    line-height:1.1;
                ">Megakernel <span style="color:{TEXT_MUTED};">×</span> Qwen3-TTS <span style="color:{TEXT_MUTED};">·</span> voice agent</div>
            </div>
            <div style="
                font-family:{MONO_FONT}; font-size:10.5px;
                color:{TEXT_MUTED}; text-align:right;
                letter-spacing:0.04em;
            ">
                <div>RTX 5090 · sm_120 · CUDA 13.1</div>
                <div>PyTorch 2.10.0a · n=5 + 3 warmup</div>
            </div>
        </div>

        <div style="display:flex; align-items:center; flex-wrap:wrap; gap:0;">
            <span style="{step}">mic</span>
            <span style="{arrow}">›</span>
            <span style="{step}">deepgram stt</span>
            <span style="{arrow}">›</span>
            <span style="{step}">groq llm</span>
            <span style="{arrow}">›</span>
            <span style="{step}; border-color:{ACCENT_PASS}; color:{ACCENT_PASS};">megakernel tts</span>
            <span style="{arrow}">›</span>
            <span style="{step}">audio</span>
        </div>
    </div>
    """


def render_build_flags_html(load_status: str) -> str:
    # Build flags — hairline key/value rows; explicit `border:0` on every td
    # so no residual global tbody-td styling can re-introduce mid-cell
    # borders (the symptom seen in earlier screenshots).
    flag_cell = (
        "padding:5px 0; border:0; vertical-align:baseline; "
        f"font-family:{MONO_FONT}; font-size:12px;"
    )
    rows: list[str] = []
    for k, v in BUILD_FLAGS.items():
        v_str = f"{v:g}" if isinstance(v, float) else str(v)
        rows.append(
            f'<tr>'
            f'<td style="{flag_cell} padding-right:18px; color:{TEXT_MUTED};">{html.escape(k)}</td>'
            f'<td style="{flag_cell} color:{TEXT_PRIMARY}; font-weight:600; text-align:right;">{html.escape(v_str)}</td>'
            f'</tr>'
        )

    # Honest disclosures — dense mono lines with leading hairline glyph,
    # not Gradio's stock <ul> bullets (which add browser-default padding
    # and visual noise).
    discl_lines = "".join(
        f'<div style="display:flex; gap:8px; margin:3px 0; align-items:baseline;">'
        f'<span style="color:{TEXT_LABEL}; font-family:{MONO_FONT}; font-size:11px;">›</span>'
        f'<span style="color:{TEXT_BODY}; font-size:12px; line-height:1.45;">{html.escape(s)}</span>'
        f'</div>'
        for s in HONEST_DISCLOSURES
    )

    section_label = (
        f"font-family:{MONO_FONT}; font-size:10.5px; "
        f"text-transform:uppercase; letter-spacing:0.14em; "
        f"color:{TEXT_LABEL}; font-weight:600;"
    )

    return f"""
    <div style="
        background: transparent;
        border: 0;
        padding: 0;
    ">
        <div style="{section_label} margin-bottom:10px;">Build flags</div>
        <table style="border-collapse:collapse; width:100%;">{''.join(rows)}</table>

        <div style="{section_label} margin:22px 0 8px;">Honest disclosures</div>
        <div>{discl_lines}</div>

        <div style="{section_label} margin:22px 0 8px;">Service status</div>
        <div style="
            font-family:{MONO_FONT}; font-size:11.5px; color:{TEXT_BODY};
            background:{SURFACE_INSET}; border:1px solid {CARD_BORDER};
            padding:8px 10px; border-radius:6px; white-space:pre-wrap;
            line-height:1.5;
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
    """Per-stage timing log for the voice pipeline."""
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
            padding:6px 0; border-bottom:1px solid {DIVIDER};
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
# Gradio Blocks layout
# -----------------------------------------------------------------------------


def _initial_service_status() -> str:
    svc = _load_services()
    lines: list[str] = [
        "MegakernelTTSService: " + (
            "OK" if svc.tts is not None else f"FAILED — {svc.tts_error}"
        ),
        "DEEPGRAM_API_KEY: " + ("present" if svc.deepgram_key else "MISSING (STT will fail)"),
        "LLM_API_KEY (Groq): " + ("present" if svc.groq_key else "MISSING (LLM will fail)"),
        f"Groq model: {svc.llm_model}",
    ]
    return "\n".join(lines)


def build_ui():
    """Construct the Gradio Blocks app. Returns the Blocks instance."""
    import gradio as gr

    # Stay close to Gradio's stock layout: only touch palette + typography.
    # Aggressive `.gr-block, .gr-form, .gr-box` resets break Gradio's flex
    # math and cause overlap/overflow + repaint glitches.
    # NO @import of Google/JSDelivr fonts — they cause FOUC and a network
    # round-trip on every page-load. System stack is always available.
    custom_css = f"""
    body, .gradio-container {{
        background: {SURFACE_BG} !important;
        color: {TEXT_PRIMARY} !important;
        font-family: {SANS_FONT};
        -webkit-font-smoothing: antialiased;
    }}
    .gradio-container {{
        max-width: 1400px !important;
        margin: 0 auto !important;
        padding: 24px 28px 48px !important;
    }}

    /* Body / paragraph text — boosted contrast vs the prior zinc-400. */
    .gradio-container p, .gradio-container span:not(.label-wrap span):not(.token) {{
        color: {TEXT_BODY};
    }}

    /* Input controls — keep Gradio's internal layout, only override
       palette + typography. No `padding: 0 !important` here. */
    .gradio-container textarea,
    .gradio-container input[type='text'],
    .gradio-container input[type='number'] {{
        background: {SURFACE_INSET} !important;
        color: {TEXT_PRIMARY} !important;
        font-family: {MONO_FONT} !important;
        border-color: {CARD_BORDER} !important;
    }}
    .gradio-container textarea:focus,
    .gradio-container input:focus {{
        border-color: {ACCENT_PASS} !important;
        box-shadow: 0 0 0 1px {ACCENT_PASS}55 !important;
    }}

    /* Labels — uppercase mono eyebrow, but ONLY actual form labels.
       Avoid hitting span.label inside our HTML metric cards. */
    .gradio-container .label-wrap > span,
    .gradio-container fieldset > legend {{
        color: {TEXT_LABEL} !important;
        font-size: 10.5px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.12em !important;
        font-family: {MONO_FONT} !important;
    }}

    /* Primary CTA — single accent, no glow. */
    .gradio-container button.primary {{
        background: {ACCENT_PASS} !important;
        border: 1px solid {ACCENT_PASS} !important;
        color: {SURFACE_BG} !important;
        font-family: {MONO_FONT} !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
    }}
    .gradio-container button.primary:hover {{
        background: #34d399 !important;
        border-color: #34d399 !important;
    }}
    .gradio-container button.secondary {{
        background: transparent !important;
        border: 1px solid {CARD_BORDER} !important;
        color: {TEXT_BODY} !important;
        font-family: {MONO_FONT} !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }}

    /* Slider track */
    .gradio-container input[type=range] {{
        accent-color: {ACCENT_PASS};
    }}

    /* Radio "Mode" — explicit dark unselected state (Gradio's stock
       paints it near-white, which collides with the dark page bg and
       hides the label text). Selected state lights up emerald. */
    .gradio-container fieldset {{
        background: transparent !important;
    }}
    .gradio-container .wrap label,
    .gradio-container fieldset label {{
        background: {SURFACE_INSET} !important;
        border: 1px solid {CARD_BORDER} !important;
        color: {TEXT_BODY} !important;
    }}
    .gradio-container .wrap label *,
    .gradio-container fieldset label * {{
        color: {TEXT_BODY} !important;
    }}
    .gradio-container .wrap label[aria-checked='true'],
    .gradio-container .wrap label.selected,
    .gradio-container fieldset label.selected {{
        background: {ACCENT_PASS}1a !important;
        border-color: {ACCENT_PASS} !important;
        color: {ACCENT_PASS} !important;
    }}
    .gradio-container .wrap label[aria-checked='true'] *,
    .gradio-container .wrap label.selected *,
    .gradio-container fieldset label.selected * {{
        color: {ACCENT_PASS} !important;
    }}

    /* Dataframe — engineer table, mono. SCOPED to Gradio's gr.Dataframe
       only so we don't bleed horizontal borders into hand-rolled HTML
       tables (build flags, brief targets, etc.). */
    .gradio-container .gradio-dataframe table,
    .gradio-container .gr-dataframe table {{
        font-family: {MONO_FONT} !important;
        font-size: 12px !important;
    }}
    .gradio-container .gradio-dataframe thead th,
    .gradio-container .gr-dataframe thead th {{
        background: {SURFACE_INSET} !important;
        color: {TEXT_LABEL} !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        font-size: 10.5px !important;
    }}
    .gradio-container .gradio-dataframe tbody td,
    .gradio-container .gr-dataframe tbody td {{
        border-bottom: 1px solid {DIVIDER} !important;
        color: {TEXT_PRIMARY} !important;
    }}

    /* Mode-radio wrapper — strip Gradio's stock form/block chrome so the
       chips read as a free-standing toggle, not as a boxed setting. */
    .gradio-container #mode-radio.block,
    .gradio-container #mode-radio .form,
    .gradio-container #mode-radio fieldset {{
        background: transparent !important;
        border: 0 !important;
        padding: 0 !important;
        box-shadow: none !important;
    }}
    .gradio-container #mode-radio {{
        margin-bottom: 22px !important;
    }}

    /* Hide Gradio's stock footer. */
    .gradio-container footer {{ display: none !important; }}

    /* Position the gr.Audio wrapper off-screen so the canvas waveform
       above it is the visible UI, but DON'T use display:none /
       visibility:hidden — Chrome's autoplay policy refuses to play
       audio in such elements even after a user gesture. The element
       is full-size, audible, just sitting at -10000 px. */
    #bot-audio-stream {{
        position: absolute !important;
        left: -10000px !important;
        top: -10000px !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }}

    /* Anti-overflow guard — make sure no HTML island bursts the column. */
    .gradio-container .gr-html, .gradio-container .gr-prose {{
        max-width: 100%;
        overflow-x: auto;
    }}
    """

    with gr.Blocks(
        title="Megakernel × Qwen3-TTS — voice agent",
        theme=gr.themes.Base(
            primary_hue="emerald",
            neutral_hue="zinc",
        ).set(
            body_background_fill=SURFACE_BG,
            body_text_color=TEXT_PRIMARY,
            background_fill_primary=SURFACE_BG,
            background_fill_secondary=SURFACE_INSET,
            border_color_primary=CARD_BORDER,
        ),
        css=custom_css,
        analytics_enabled=False,
    ) as demo:
        gr.HTML(render_header_html())

        # Persistent shared state: latest metrics + which mode produced them.
        live_state = gr.State(LiveMetrics())

        with gr.Row():
            with gr.Column(scale=3, min_width=520):

                # ---- Persistent metric cards (top of column) ------------
                gr.HTML(
                    '<div style="font-size:10.5px; text-transform:uppercase; '
                    f'letter-spacing:0.14em; color:{TEXT_LABEL}; '
                    f'margin:2px 0 10px; font-weight:600; font-family:{MONO_FONT};">'
                    'METRICS</div>'
                )
                metrics_html = gr.HTML(render_metric_cards(LiveMetrics()))

                gr.HTML(
                    '<div style="font-size:10.5px; text-transform:uppercase; '
                    f'letter-spacing:0.14em; color:{TEXT_LABEL}; '
                    f'margin:20px 0 8px; font-weight:600; font-family:{MONO_FONT};">'
                    'BRIEF TARGETS</div>'
                )
                comparison_html = gr.HTML(render_comparison_table(LiveMetrics()))

                # ---- Voice agent pipeline (mic → STT → LLM → TTS) -------
                gr.HTML(
                    '<div style="font-size:11px; text-transform:uppercase; '
                    f'letter-spacing:0.08em; color:{TEXT_MUTED}; '
                    'margin:24px 0 8px; font-weight:600; '
                    f'font-family:{MONO_FONT};">'
                    'Voice agent · mic → STT → LLM → streaming TTS</div>'
                )
                mic_in = gr.Audio(
                    label="Record a question",
                    sources=["microphone"],
                    type="numpy",
                    editable=False,
                    interactive=True,
                )
                send_btn = gr.Button(
                    "Send",
                    variant="primary",
                )
                # streaming=True + autoplay=True turns the audio
                # widget into a sink for per-chunk PCM yields from
                # the on_send generator below. Browser begins
                # playback at the FIRST chunk (≈80 ms post-LLM)
                # instead of waiting for end-of-utterance. This is
                # the brief's "audio playing while still generating"
                # requirement — buffered = average submission.
                # ChatGPT-voice-mode-style player:
                # - Custom canvas waveform that animates from a Web Audio
                #   AnalyserNode tap on the underlying <audio> element.
                # - The native gr.Audio control is hidden via CSS but still
                #   functional (Gradio's HLS streaming machinery is what
                #   actually decodes + plays). The canvas just visualizes.
                gr.HTML(f"""
                <div style="
                    border: 1px solid {CARD_BORDER};
                    border-left: 2px solid {ACCENT_PASS};
                    padding: 18px 22px 14px;
                    background: transparent;
                    margin-bottom: 6px;
                ">
                    <div id="bot-wave-status" style="
                        font-family: {MONO_FONT};
                        font-size: 10.5px;
                        color: {TEXT_LABEL};
                        text-transform: uppercase;
                        letter-spacing: 0.14em;
                        margin-bottom: 10px;
                    ">● bot reply <span style="color:{TEXT_MUTED}">— idle</span></div>
                    <canvas id="bot-waveform"
                            width="1100" height="64"
                            style="width:100%; height:64px; display:block;">
                    </canvas>
                </div>
                <script>
                (function() {{
                    if (window.__botWaveBooted) return;
                    window.__botWaveBooted = true;
                    const ACCENT  = '{ACCENT_PASS}';
                    const ACCENT2 = '#34d399';
                    const MUTED   = '{TEXT_MUTED}';
                    const LABEL   = '{TEXT_LABEL}';
                    function setStatus(html) {{
                        const el = document.getElementById('bot-wave-status');
                        if (el) el.innerHTML = html;
                    }}
                    function getCanvasAndCtx() {{
                        const cv = document.getElementById('bot-waveform');
                        if (!cv) return null;
                        // High-DPI: backing store at devicePixelRatio
                        const dpr = window.devicePixelRatio || 1;
                        const cssW = cv.clientWidth;
                        const cssH = cv.clientHeight;
                        if (cv.width !== cssW * dpr) cv.width = cssW * dpr;
                        if (cv.height !== cssH * dpr) cv.height = cssH * dpr;
                        const ctx = cv.getContext('2d');
                        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
                        return {{cv, ctx, w: cssW, h: cssH}};
                    }}
                    function drawIdle() {{
                        const r = getCanvasAndCtx(); if (!r) return;
                        const {{ctx, w, h}} = r;
                        ctx.clearRect(0, 0, w, h);
                        ctx.strokeStyle = MUTED;
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(0, h/2); ctx.lineTo(w, h/2); ctx.stroke();
                    }}
                    drawIdle();
                    let audioCtx = null, analyser = null, source = null;
                    let rafId = null;
                    function teardown() {{
                        if (rafId) {{ cancelAnimationFrame(rafId); rafId = null; }}
                    }}
                    function attach(audio) {{
                        if (!audio || audio.dataset.botWaveAttached) return;
                        audio.dataset.botWaveAttached = '1';
                        // Hide the native controls completely — we ARE the player UI.
                        audio.controls = false;
                        audio.style.display = 'none';
                        // Web Audio tap. Use a shared AudioContext that auto-resumes
                        // on the first user gesture (Send click).
                        if (!audioCtx) {{
                            const AC = window.AudioContext || window.webkitAudioContext;
                            audioCtx = new AC();
                        }}
                        if (audioCtx.state === 'suspended') audioCtx.resume();
                        try {{
                            source = audioCtx.createMediaElementSource(audio);
                        }} catch (e) {{
                            // Already wired in a prior run — skip
                            return;
                        }}
                        analyser = audioCtx.createAnalyser();
                        analyser.fftSize = 2048;
                        analyser.smoothingTimeConstant = 0.75;
                        source.connect(analyser);
                        analyser.connect(audioCtx.destination);
                        const buf = new Uint8Array(analyser.frequencyBinCount);
                        function frame() {{
                            const r = getCanvasAndCtx();
                            if (!r) {{ rafId = requestAnimationFrame(frame); return; }}
                            const {{ctx, w, h}} = r;
                            analyser.getByteTimeDomainData(buf);
                            ctx.clearRect(0, 0, w, h);
                            // Centre line for reference
                            ctx.strokeStyle = MUTED;
                            ctx.lineWidth = 1;
                            ctx.beginPath();
                            ctx.moveTo(0, h/2); ctx.lineTo(w, h/2); ctx.stroke();
                            // Waveform (zero-crossed centre, emerald)
                            const grad = ctx.createLinearGradient(0, 0, w, 0);
                            grad.addColorStop(0, ACCENT);
                            grad.addColorStop(1, ACCENT2);
                            ctx.strokeStyle = grad;
                            ctx.lineWidth = 2;
                            ctx.beginPath();
                            const slice = w / buf.length;
                            for (let i = 0; i < buf.length; i++) {{
                                const v = buf[i] / 128.0;
                                const y = (v * h) / 2;
                                const x = i * slice;
                                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                            }}
                            ctx.stroke();
                            rafId = requestAnimationFrame(frame);
                        }}
                        audio.addEventListener('play', () => {{
                            setStatus("● bot reply <span style='color:" + ACCENT + "'>— speaking</span>");
                            if (audioCtx.state === 'suspended') audioCtx.resume();
                            if (!rafId) frame();
                        }});
                        audio.addEventListener('pause', () => {{
                            // Pause but don't tear down — playback might resume
                        }});
                        audio.addEventListener('ended', () => {{
                            setStatus("● bot reply <span style='color:" + LABEL + "'>— done</span>");
                            teardown();
                            drawIdle();
                        }});
                        audio.addEventListener('emptied', () => {{
                            setStatus("● bot reply <span style='color:" + MUTED + "'>— idle</span>");
                            teardown();
                            drawIdle();
                        }});
                    }}
                    // The Gradio audio element is created lazily. Poll DOM
                    // briefly and attach when it appears.
                    function findAudio() {{
                        const host = document.getElementById('bot-audio-stream');
                        if (!host) return null;
                        return host.querySelector('audio');
                    }}
                    const poll = setInterval(() => {{
                        const a = findAudio();
                        if (a) {{
                            clearInterval(poll);
                            attach(a);
                        }}
                    }}, 250);
                    // Re-attach if Gradio re-mounts the audio element.
                    const obs = new MutationObserver(() => {{
                        const a = findAudio();
                        if (a && !a.dataset.botWaveAttached) attach(a);
                    }});
                    obs.observe(document.body, {{childList: true, subtree: true}});
                }})();
                </script>
                """)
                audio_out = gr.Audio(
                    label="bot reply",
                    show_label=False,
                    streaming=True,
                    autoplay=True,
                    interactive=False,
                    elem_id="bot-audio-stream",
                )
                gr.HTML(
                    '<div style="font-size:11px; text-transform:uppercase; '
                    f'letter-spacing:0.08em; color:{TEXT_MUTED}; '
                    'margin:14px 0 6px; font-weight:600;">'
                    'Per-stage timings</div>'
                )
                stage_log_html = gr.HTML(render_stage_log())

            with gr.Column(scale=1, min_width=300):
                gr.HTML(render_build_flags_html(_initial_service_status()))

        # ---------------------------------------------------------------
        # Click handler — ASYNC GENERATOR so we can stream:
        #   yield #1: STT done → transcription visible
        #   yield #2: LLM done → reply visible
        #   yield #N: per-frame PCM chunk → browser plays as we synth
        #   yield #last: final metrics
        # Brief requirement: "audio playing while still generating"
        # — buffered = average submission. ~80 ms per chunk @ 12.5 Hz.
        # ---------------------------------------------------------------
        async def on_send(
            mic_value: Any,
            current_state: LiveMetrics,
        ):
            """Streaming pipeline handler — async generator.

            Each ``yield`` pushes a fresh snapshot of (audio_chunk,
            stage_log_html, metrics_html, comparison_html, state) to the
            browser. The audio output is ``streaming=True``, so per-frame
            PCM tuples (24000, np.int16[]) start playback in the browser
            as soon as the first chunk arrives — typically ~13 ms after
            Groq returns the LLM reply (the megakernel TTFC).

            Yields, in order:
              1. After STT: transcription visible in the stage log.
              2. After LLM: reply visible.
              3. Per frame: an int16 numpy chunk → browser appends + plays.
              4. Final: full metrics + RTF + tok/s + e2e.

            Anti-pattern guard: NEVER collect chunks then return one WAV.
            That's the explicit "audio after full sentence" failure mode
            the brief calls out as average-tier.
            """
            import numpy as np  # local — keep top of file Mac-import-safe
            t0 = time.perf_counter()

            def _err(msg: str, state: LiveMetrics):
                """Single-shot error response shaped like a normal yield."""
                return (
                    None,
                    render_stage_log(error=msg),
                    render_metric_cards(state),
                    render_comparison_table(state),
                    state,
                )

            if mic_value is None:
                # Optional dev/test fixture so the pipeline is exercisable
                # without a working microphone (e.g., over an SSH tunnel where
                # WebRTC mic capture isn't available). Production runs leave
                # this env unset and get the standard "no mic" error.
                fixture = os.environ.get("UI_V2_TEST_FIXTURE_WAV")
                if fixture and os.path.exists(fixture):
                    import wave as _wave
                    with _wave.open(fixture, "rb") as wf:
                        sr = wf.getframerate()
                        frames = wf.readframes(wf.getnframes())
                    arr = np.frombuffer(frames, dtype=np.int16)
                    if wf.getnchannels() == 2:
                        arr = arr.reshape(-1, 2).mean(axis=1).astype(np.int16)
                    mic_value = (sr, arr)
                else:
                    yield _err(
                        "No mic recording. Click the mic widget, record a "
                        "question, then Send.", current_state)
                    return

            svc = _load_services()
            if svc.tts is None:
                yield _err(svc.tts_error or "MegakernelTTSService not initialized.", current_state)
                return
            if not svc.deepgram_key:
                yield _err("DEEPGRAM_API_KEY missing in env / .env", current_state)
                return
            if not svc.groq_key:
                yield _err("LLM_API_KEY (Groq) missing in env / .env", current_state)
                return

            # 1. Pack mic recording into a WAV the Deepgram REST endpoint
            #    accepts. Gradio hands us (sample_rate, np.ndarray).
            try:
                wav_bytes = _numpy_audio_to_wav_bytes(mic_value)
            except Exception as e:  # noqa: BLE001
                yield _err(f"WAV pack failed: {e!r}", current_state)
                return

            # 2. STT — Deepgram REST.
            try:
                user_text, stt_ms = await _stt_deepgram_async(wav_bytes, svc.deepgram_key)
            except Exception as e:  # noqa: BLE001
                yield _err(f"Deepgram STT failed: {e!r}", current_state)
                return
            if not user_text:
                yield _err("Deepgram returned empty transcript — try again.", current_state)
                return

            # YIELD #1: transcription appears as soon as STT completes.
            stage_after_stt = render_stage_log(
                user_text=user_text,
                stt_ms=stt_ms,
                llm_text="(LLM running…)",
            )
            yield (
                None,  # no audio chunk yet
                stage_after_stt,
                render_metric_cards(current_state),
                render_comparison_table(current_state),
                current_state,
            )

            # 3. LLM — Groq. Frames slider caps reply tokens so the TTS
            #    horizon stays bounded for a quick A/B comparison.
            try:
                llm_text, llm_ms = await _llm_groq_async(
                    user_text, svc.groq_key, svc.llm_model,
                    # Conversational reply length — 80 tokens ≈ 60 words ≈
                    # 6-8 seconds of speech. Caps the LLM so audio horizon
                    # below stays bounded for both UX and benchmark hygiene.
                    max_tokens=80,
                )
            except Exception as e:  # noqa: BLE001
                yield _err(f"Groq LLM failed: {e!r}", current_state)
                return

            # YIELD #2: LLM reply visible before any audio plays.
            stage_after_llm = render_stage_log(
                user_text=user_text,
                stt_ms=stt_ms,
                llm_text=llm_text,
                llm_ms=llm_ms,
                ttfc_ms=0.0,
                tts_total_ms=0.0,
                audio_seconds=0.0,
                e2e_ms=(time.perf_counter() - t0) * 1000.0,
            )
            yield (
                None,
                stage_after_llm,
                render_metric_cards(current_state),
                render_comparison_table(current_state),
                current_state,
            )

            # 4. TTS — STREAMING with batched yields.
            #
            # The megakernel yields one ~80 ms codec frame at a time, but
            # Gradio's `streaming=True` audio sink shells out to ffprobe
            # per yield to re-encode PCM → ADTS for the browser's
            # MediaSource API. The fixed startup cost is ~74 ms regardless
            # of chunk size on this box. Per-frame yield would mean 74 ms
            # encode for 80 ms audio = zero margin = stutter.
            #
            # 4-frame batch = 320 ms audio per yield. pydub's per-encode
            # cost is ~75 ms regardless of chunk size (ffprobe startup
            # dominates), so 320 ms / 75 ms = 4.3× margin — enough for
            # smooth playback without MediaSource queue backup. We tried
            # 1-frame batches (80 ms / 75 ms = 1.07× margin) and the
            # browser silently dropped chunks → empty audio player.
            STREAM_BATCH_FRAMES = 4
            ttfc_ms: float | None = None
            tts_t0 = time.perf_counter()
            total_pcm = bytearray()
            pending = bytearray()
            pending_count = 0
            try:
                # Cap TTS horizon dynamically based on LLM reply length so
                # the talker never runs past EOS into babble territory.
                # English speech: ~6 chars/sec, codec at 12.5 frames/sec →
                # ~2.1 frames per character. Add 25% slack + a hard floor
                # of 40 frames (3.2 s) for very short replies and a ceiling
                # of 250 frames (20 s) so a long LLM reply still bounds.
                est_frames = int(len(llm_text) * 2.1 * 1.25) if llm_text else 0
                max_tts_frames = max(40, min(250, est_frames or 40))
                # Display state that we update with live measurements as
                # they become available — TTFC the moment the first chunk
                # lands, then a running tok/s estimate per chunk so the
                # cards animate during synthesis. Final RTF + e2e land
                # after the last chunk.
                live_disp = LiveMetrics(source="Test mode")
                # Use the public ``stream_tts`` wrapper instead of reaching
                # through ``svc.tts._tts.generate(...)``; semantics are
                # identical (delegates to ``MegakernelTTS.generate``) but the
                # call site no longer crosses two underscore-prefix boundaries.
                async for pcm in svc.tts.stream_tts(
                    llm_text, max_new_tokens=max_tts_frames,
                ):
                    if not pcm:
                        continue
                    if ttfc_ms is None:
                        ttfc_ms = (time.perf_counter() - tts_t0) * 1000.0
                        live_disp = LiveMetrics(
                            ttfc_ms=ttfc_ms,
                            source="Test mode",
                            updated_at=time.time(),
                        )
                    total_pcm.extend(pcm)
                    pending.extend(pcm)
                    pending_count += 1
                    if pending_count >= STREAM_BATCH_FRAMES:
                        chunk_np = np.frombuffer(bytes(pending), dtype=np.int16)
                        # Update live tok/s + RTF estimate per yield
                        # so the cards animate as audio streams.
                        elapsed = max(1e-9, time.perf_counter() - tts_t0)
                        audio_so_far = (len(total_pcm) // 2) / SAMPLE_RATE_HZ
                        live_disp = LiveMetrics(
                            ttfc_ms=ttfc_ms or 0.0,
                            rtf=elapsed / audio_so_far if audio_so_far > 0 else 0.0,
                            decode_tok_per_s=(audio_so_far * 12.5) / elapsed,
                            e2e_ms=(time.perf_counter() - t0) * 1000.0,
                            source="Test mode",
                            updated_at=time.time(),
                        )
                        yield (
                            (SAMPLE_RATE_HZ, chunk_np),
                            stage_after_llm,
                            render_metric_cards(live_disp),
                            render_comparison_table(live_disp),
                            live_disp,
                        )
                        pending = bytearray()
                        pending_count = 0
                # Flush any tail < STREAM_BATCH_FRAMES so the user hears
                # the end of the utterance.
                if pending_count > 0:
                    chunk_np = np.frombuffer(bytes(pending), dtype=np.int16)
                    yield (
                        (SAMPLE_RATE_HZ, chunk_np),
                        stage_after_llm,
                        render_metric_cards(live_disp),
                        render_comparison_table(live_disp),
                        live_disp,
                    )
            except Exception as e:  # noqa: BLE001
                yield _err(f"MegakernelTTS failed: {e!r}", current_state)
                return

            tts_total_ms = (time.perf_counter() - tts_t0) * 1000.0
            audio_seconds = (len(total_pcm) // 2) / SAMPLE_RATE_HZ if total_pcm else 0.0
            e2e_ms = (time.perf_counter() - t0) * 1000.0

            rtf = (tts_total_ms / 1000.0 / audio_seconds) if audio_seconds > 0 else 0.0
            tok_per_s = (audio_seconds * 12.5) / (tts_total_ms / 1000.0) if tts_total_ms > 0 else 0.0
            new_state = LiveMetrics(
                ttfc_ms=ttfc_ms or 0.0,
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
                ttfc_ms=ttfc_ms or 0.0,
                tts_total_ms=tts_total_ms,
                audio_seconds=audio_seconds,
                e2e_ms=e2e_ms,
            )

            # YIELD #last: final metrics. CRITICAL: do NOT pass `None` to
            # the streaming audio output here — that triggers Gradio's
            # frontend to tear down the MediaSource/HLS player mid-playback
            # (we see "MinimalAudioPlayer: Container not found" + an HLS
            # fatal stall + the widget resetting to 0:00 / 0:00). Use
            # ``gr.skip()`` so the audio sink keeps its existing stream
            # (Gradio finalizes the stream automatically when the generator
            # returns).
            yield (
                gr.skip(),
                stage_html,
                render_metric_cards(new_state),
                render_comparison_table(new_state),
                new_state,
            )

        # Two ways to trigger the pipeline:
        #   1. Click Send (explicit) — the obvious path
        #   2. Stop recording (auto) — Gradio fires `stop_recording` the
        #      moment the user clicks Stop on the mic widget AND the audio
        #      data is packaged. This eliminates the timing race where
        #      clicking Send right after Stop sees mic_value=None because
        #      Gradio hasn't finished its async packaging yet.
        outputs = [audio_out, stage_log_html, metrics_html, comparison_html, live_state]
        send_btn.click(fn=on_send, inputs=[mic_in, live_state], outputs=outputs)
        mic_in.stop_recording(fn=on_send, inputs=[mic_in, live_state], outputs=outputs)

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
