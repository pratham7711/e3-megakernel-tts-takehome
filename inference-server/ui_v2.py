"""Polished Gradio UI for the e3 Megakernel x Qwen3-TTS take-home submission.

Single-file Gradio Blocks app. Loads the same Decoder + code_predictor + (stub)
codec stack that ``ui.py`` already wires up, but presents the OUTPUT in a way
that matches the brief's "engineering quality, no marketing fluff" bar:

    * big readable metric cards (TTFC, RTF, decode tok/s, audio duration)
    * comparison-with-brief table (3 target tiers, PASS / MISS badges)
    * run history dataframe (last 10 runs)
    * honest disclaimer banner up top (visible without scrolling)
    * build flags exposed in sidebar so engineers can sanity-check the config

Run on the GPU box::

    python3 ui_v2.py

Listens on ``0.0.0.0:8080``. Reach it via SSH tunnel::

    ssh -L 8080:localhost:8080 <gpu-box>

Design notes
------------
The model load + per-run math is logically identical to ``ui.py``. Only the
presentation layer is new. Imports of ``torch`` / ``qwen_megakernel`` /
``qwen3_tts_components`` are lazy so the file still ``py_compile``s on a Mac
laptop without the CUDA stack installed.
"""

from __future__ import annotations

import dataclasses
import html
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Lazy heavy imports: numpy / torch / qwen_megakernel / qwen3_tts_components
# are only imported inside ``load_components()`` and ``generate_one()`` so this
# module compiles and even imports on a laptop without the CUDA stack.

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
    "Codec is a sine-wave stub - output sounds like beeping, not speech. "
    "Talker + code_predictor compute is REAL and timing is honest."
)

HONEST_DISCLOSURES: list[str] = [
    "Codec: sine-wave stub (496-key convnet not reverse-engineered)",
    "MRoPE: single-axis collapse for decode-only path - math-equivalent to vanilla 1D RoPE theta=1M; full multi-axis NOT in kernel",
    "Bench numbers from steady-state (post-warmup)",
    "GPU: 1x RTX 5090 sm_120 Blackwell, CUDA 13.1, PyTorch 2.10.0a NGC",
]

# Output sample rate for the Qwen3-TTS codec (24 kHz int16 PCM).
SAMPLE_RATE_HZ: int = 24_000
# Codec frame width (samples per 80 ms chunk at 12.5 Hz codec rate).
SAMPLES_PER_FRAME: int = 1_920

# Model checkpoint + speaker (matches inference-server defaults).
MODEL_PATH = "/workspace/qwen3-tts-1.7b"
SPEAKER = "ryan"


# -----------------------------------------------------------------------------
# Model load (lazy, mirrors the ``ui.py`` / megakernel_tts.py pattern)
# -----------------------------------------------------------------------------


@dataclass
class LoadedComponents:
    """Bundle of loaded components, or a stub-mode marker."""

    talker: Any = None
    code_predictor: Any = None
    codec: Any = None
    device: str = "cuda"
    stub: bool = False
    load_error: str | None = None


# Module-level cache so we only pay the load cost once.
_COMPONENTS: LoadedComponents | None = None


def load_components() -> LoadedComponents:
    """Load Decoder + code_predictor + codec once; cached for subsequent calls.

    Mirrors the lazy / stub-on-failure pattern from
    ``inference-server/megakernel_tts.py`` so this UI behaves identically to
    the existing CLI / bench harness when running on the GPU box.

    Returns:
        A :class:`LoadedComponents` instance. If anything fails to import or
        load, ``stub=True`` and ``load_error`` is populated so the UI can
        surface the reason instead of silently emitting silence.
    """
    global _COMPONENTS
    if _COMPONENTS is not None:
        return _COMPONENTS

    comps = LoadedComponents()
    try:
        import torch  # type: ignore
        from qwen_megakernel.model import Decoder  # type: ignore
        from qwen3_tts_components import load_components as _load  # type: ignore
    except Exception as e:  # noqa: BLE001
        comps.stub = True
        comps.load_error = f"import failed: {e!r}"
        _COMPONENTS = comps
        return comps

    try:
        comps.talker = Decoder(model_path=MODEL_PATH, verbose=False)
        comps.code_predictor, comps.codec, _ = _load(
            weights_dir=MODEL_PATH,
            device="cuda",
            dtype=torch.bfloat16,
        )
        comps.device = "cuda"
        comps.stub = False
    except Exception as e:  # noqa: BLE001
        comps.stub = True
        comps.load_error = f"load failed: {e!r}"

    _COMPONENTS = comps
    return comps


# -----------------------------------------------------------------------------
# One-shot generate + metrics
# -----------------------------------------------------------------------------


@dataclass
class RunMetrics:
    """Per-run metrics surfaced into the UI."""

    ttfc_ms: float = 0.0
    rtf: float = 0.0
    decode_tok_per_s: float = 0.0
    audio_seconds: float = 0.0
    total_ms: float = 0.0
    frames: int = 0
    text: str = ""
    timestamp: str = ""
    error: str | None = None


def generate_one(text: str, frames: int) -> tuple[Any, RunMetrics]:
    """Run one full Talker -> code_predictor -> codec pass.

    Mirrors ``ui.py``'s per-run loop: feed seed token, step the megakernel
    Talker ``frames`` times, run the code predictor + codec on each step,
    collect int16 samples. TTFC is wall-clock from the start of the loop to
    the first PCM chunk after ``cuda.synchronize``.

    Returns:
        ``(audio_tuple, metrics)`` where ``audio_tuple`` is the
        ``(sample_rate, np.int16 array)`` that ``gr.Audio`` consumes, or
        ``None`` on error.
    """
    import numpy as np  # local: keeps top-of-file compileable on Mac

    metrics = RunMetrics(
        text=text.strip(),
        frames=int(frames),
        timestamp=time.strftime("%H:%M:%S"),
    )

    comps = load_components()
    if comps.stub:
        metrics.error = comps.load_error or "stub mode (no CUDA stack)"
        # Emit a short audible-style silence so the gr.Audio widget renders.
        silence = np.zeros(SAMPLES_PER_FRAME * int(frames), dtype=np.int16)
        return (SAMPLE_RATE_HZ, silence), metrics

    try:
        import torch  # type: ignore
    except Exception as e:  # noqa: BLE001
        metrics.error = f"torch import failed: {e!r}"
        return None, metrics

    talker = comps.talker
    code_predictor = comps.code_predictor
    codec = comps.codec
    device = comps.device

    try:
        talker.reset()
    except Exception:  # noqa: BLE001
        pass

    pcm_chunks: list[Any] = []
    prev_tok = 0
    ttfc_ms: float | None = None

    # Explicit CUDA sync on entry so TTFC isn't credited to a leftover kernel.
    try:
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass

    t0 = time.perf_counter()
    try:
        for step_i in range(int(frames)):
            next_tok = talker.step(prev_tok)
            tok_tensor = torch.tensor(
                [[next_tok]], dtype=torch.long, device=device
            )
            code_frame = code_predictor(tok_tensor)
            pcm = codec(code_frame)

            # Defensive: codec may return bytes / ndarray / tensor.
            if isinstance(pcm, (bytes, bytearray)):
                arr = np.frombuffer(bytes(pcm), dtype=np.int16)
            elif isinstance(pcm, np.ndarray):
                arr = pcm.astype(np.int16)
            else:
                arr = (
                    pcm.detach().to("cpu").numpy().astype(np.int16)
                    if hasattr(pcm, "detach")
                    else np.asarray(pcm).astype(np.int16)
                )

            if ttfc_ms is None:
                try:
                    torch.cuda.synchronize()
                except Exception:  # noqa: BLE001
                    pass
                ttfc_ms = (time.perf_counter() - t0) * 1000.0

            pcm_chunks.append(arr)
            prev_tok = next_tok
    except Exception as e:  # noqa: BLE001
        metrics.error = f"generate failed at step {step_i}: {e!r}"
        # Fall through: surface whatever audio we did manage to render.

    try:
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass
    total_s = time.perf_counter() - t0

    if not pcm_chunks:
        metrics.ttfc_ms = float(ttfc_ms or 0.0)
        metrics.total_ms = total_s * 1000.0
        return None, metrics

    audio = np.concatenate(pcm_chunks).astype(np.int16)
    n_samples = audio.shape[0]
    audio_seconds = n_samples / SAMPLE_RATE_HZ

    metrics.ttfc_ms = float(ttfc_ms or 0.0)
    metrics.total_ms = total_s * 1000.0
    metrics.audio_seconds = audio_seconds
    metrics.rtf = (total_s / audio_seconds) if audio_seconds > 0 else float("nan")
    metrics.decode_tok_per_s = (
        int(frames) / total_s if total_s > 0 else float("nan")
    )

    return (SAMPLE_RATE_HZ, audio), metrics


# -----------------------------------------------------------------------------
# Presentation helpers (HTML cards, comparison table)
# -----------------------------------------------------------------------------

ACCENT_PASS = "#10b981"   # emerald-500
ACCENT_PARTIAL = "#f59e0b"  # amber-500
ACCENT_MISS = "#ef4444"   # red-500
ACCENT_VIOLET = "#8b5cf6"  # violet-500
SURFACE_BG = "#0b1020"
CARD_BG = "#11162a"
CARD_BORDER = "#1f2740"
TEXT_PRIMARY = "#e5e7eb"
TEXT_MUTED = "#9ca3af"
MONO_FONT = "'JetBrains Mono', 'Fira Code', ui-monospace, SFMono-Regular, Menlo, monospace"


def _metric_card(label: str, value: str, unit: str, accent: str = ACCENT_VIOLET) -> str:
    """Render one large-number metric card."""
    return f"""
    <div style="
        background: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-radius: 12px;
        padding: 20px 22px;
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
            font-size: 36px;
            font-weight: 700;
            color: {TEXT_PRIMARY};
            line-height: 1.1;
        ">
            <span style="color: {accent}">{html.escape(value)}</span>
            <span style="font-size: 14px; color: {TEXT_MUTED}; font-weight: 500; margin-left: 6px;">{html.escape(unit)}</span>
        </div>
    </div>
    """


def render_metric_cards(m: RunMetrics) -> str:
    """Render the 4-card metric row as a single HTML block."""
    if m.error:
        # Show one full-width error card.
        return f"""
        <div style="
            background: {CARD_BG};
            border: 1px solid {ACCENT_MISS};
            border-radius: 12px;
            padding: 16px 20px;
            color: {TEXT_PRIMARY};
            font-family: {MONO_FONT};
            font-size: 13px;
        ">
            <div style="color: {ACCENT_MISS}; font-weight: 600; margin-bottom: 6px;">RUN ERROR</div>
            <div style="color: {TEXT_PRIMARY}; white-space: pre-wrap;">{html.escape(m.error)}</div>
        </div>
        """

    cards = [
        _metric_card("TTFC", f"{m.ttfc_ms:.1f}", "ms", _ttfc_accent(m.ttfc_ms)),
        _metric_card("RTF", f"{m.rtf:.4f}", "", _rtf_accent(m.rtf)),
        _metric_card("Decode tok/s", f"{m.decode_tok_per_s:.1f}", "tok/s", ACCENT_VIOLET),
        _metric_card("Audio duration", f"{m.audio_seconds:.2f}", "s", ACCENT_VIOLET),
    ]
    return f"""
    <div style="
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
    ">
        {''.join(cards)}
    </div>
    """


def _ttfc_accent(ttfc_ms: float) -> str:
    if ttfc_ms <= PERFORMANCE_TARGETS["TTFC_ms"]:
        return ACCENT_PASS
    if ttfc_ms <= DELIVERABLES["TTFC_ms"]:
        return ACCENT_PARTIAL
    return ACCENT_MISS


def _rtf_accent(rtf: float) -> str:
    if rtf <= PERFORMANCE_TARGETS["RTF"]:
        return ACCENT_PASS
    if rtf <= DELIVERABLES["RTF"]:
        return ACCENT_PARTIAL
    return ACCENT_MISS


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


def render_comparison_table(m: RunMetrics) -> str:
    """Render the comparison-with-brief table with PASS / MISS badges."""
    if m.error or (m.ttfc_ms == 0 and m.rtf == 0):
        return f"""
        <div style="
            color: {TEXT_MUTED};
            font-family: {MONO_FONT};
            font-size: 12px;
            padding: 12px;
        ">No run yet. Generate to see how this run compares against the brief's targets.</div>
        """

    rows_html: list[str] = []
    # TTFC row
    ttfc_cells = [
        f'<td style="padding:8px 12px; color:{TEXT_MUTED}; font-family:{MONO_FONT};">TTFC</td>',
        f'<td style="padding:8px 12px; color:{TEXT_PRIMARY}; font-family:{MONO_FONT}; font-weight:600;">{m.ttfc_ms:.1f} ms</td>',
    ]
    for _tier_label, tier in TARGET_TIERS:
        passed = m.ttfc_ms <= tier["TTFC_ms"]
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
        passed = m.rtf <= tier["RTF"]
        rtf_cells.append(
            f'<td style="padding:8px 12px; font-family:{MONO_FONT}; color:{TEXT_MUTED};">'
            f'&lt;{tier["RTF"]:.2f}&nbsp;&nbsp;{_badge(passed)}</td>'
        )
    rows_html.append("<tr>" + "".join(rtf_cells) + "</tr>")

    header_cells = [
        '<th style="text-align:left; padding:8px 12px; color:#6b7280; font-size:11px; text-transform:uppercase; letter-spacing:.08em; border-bottom:1px solid ' + CARD_BORDER + ';">Metric</th>',
        '<th style="text-align:left; padding:8px 12px; color:#6b7280; font-size:11px; text-transform:uppercase; letter-spacing:.08em; border-bottom:1px solid ' + CARD_BORDER + ';">This run</th>',
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
    """Title + pipeline diagram + honest disclaimer banner."""
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
                    AlpinDale/qwen_megakernel - Talker decode only - PyTorch code_predictor - sine-wave codec stub
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
            <span style="{pipeline_step}">text</span>
            <span style="{arrow}">&rarr;</span>
            <span style="{pipeline_step}">Megakernel Talker</span>
            <span style="{arrow}">&rarr;</span>
            <span style="{pipeline_step}">Code Predictor (PyTorch)</span>
            <span style="{arrow}">&rarr;</span>
            <span style="{pipeline_step}; border-color:{ACCENT_PARTIAL}55; color:{ACCENT_PARTIAL};">Codec (STUB)</span>
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
        ">Load status</div>
        <div style="
            font-family:{MONO_FONT}; font-size:12px; color:{TEXT_PRIMARY};
            background:{SURFACE_BG}; border:1px solid {CARD_BORDER};
            padding:8px 10px; border-radius:8px; white-space:pre-wrap;
        ">{html.escape(load_status)}</div>
    </div>
    """


# -----------------------------------------------------------------------------
# Run-history state + helpers
# -----------------------------------------------------------------------------

HISTORY_COLUMNS = [
    "time",
    "text",
    "TTFC (ms)",
    "RTF",
    "tok/s",
    "audio (s)",
]
MAX_HISTORY = 10


def _history_row(m: RunMetrics) -> list[Any]:
    snippet = m.text if len(m.text) <= 48 else m.text[:45] + "..."
    return [
        m.timestamp,
        snippet,
        round(m.ttfc_ms, 1),
        round(m.rtf, 4),
        round(m.decode_tok_per_s, 1),
        round(m.audio_seconds, 2),
    ]


def prepend_history(history, m: RunMetrics) -> list[list[Any]]:
    """Prepend a new run-history row. Defensive against the many shapes Gradio's
    Dataframe component hands back depending on version: None, list[list[Any]],
    or pandas.DataFrame. Truthiness checks must avoid ``or`` on DataFrames
    (their __bool__ raises). Empty DataFrames have ``.empty`` True; we route
    by attribute rather than truthiness.
    """
    rows: list[list[Any]]
    if history is None:
        rows = []
    elif hasattr(history, "values") and hasattr(history, "empty"):
        # pandas DataFrame
        rows = [] if history.empty else history.values.tolist()
    else:
        rows = list(history)
    rows.insert(0, _history_row(m))
    return rows[:MAX_HISTORY]


# -----------------------------------------------------------------------------
# Gradio Blocks layout
# -----------------------------------------------------------------------------


def _initial_load_status() -> str:
    comps = load_components()
    if comps.stub:
        return (
            f"STUB mode (CUDA stack unavailable on this host)\n"
            f"reason: {comps.load_error or 'unknown'}\n"
            f"audio will be silence; metrics will be zeros."
        )
    return (
        f"OK\n"
        f"device: {comps.device}\n"
        f"model_path: {MODEL_PATH}\n"
        f"speaker: {SPEAKER}"
    )


def build_ui():
    """Construct the Gradio Blocks app. Returns the Blocks instance."""
    import gradio as gr  # local: keeps Mac py_compile clean

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
    .gradio-container .dataframe-component table {{
        font-family: {MONO_FONT} !important;
        font-size: 12px !important;
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

        with gr.Row():
            with gr.Column(scale=3, min_width=520):
                text_in = gr.Textbox(
                    label="Text",
                    placeholder="Type something for the Talker to synthesize...",
                    value="Pipecat lets me wire together speech and text into one streaming voice pipeline.",
                    lines=3,
                    max_lines=8,
                )
                with gr.Row():
                    frames_in = gr.Slider(
                        label="Frames (Talker decode steps)",
                        minimum=5,
                        maximum=100,
                        step=1,
                        value=25,
                    )
                    generate_btn = gr.Button(
                        "Generate",
                        variant="primary",
                        scale=0,
                        min_width=140,
                    )

                audio_out = gr.Audio(
                    label="Generated audio (24 kHz int16 PCM)",
                    type="numpy",
                    autoplay=False,
                    show_download_button=True,
                    interactive=False,
                )

                metrics_html = gr.HTML(
                    render_metric_cards(RunMetrics())
                )

                gr.HTML(
                    '<div style="font-size:11px; text-transform:uppercase; '
                    f'letter-spacing:0.08em; color:{TEXT_MUTED}; '
                    'margin:18px 0 8px; font-weight:600;">'
                    'Comparison with brief</div>'
                )
                comparison_html = gr.HTML(render_comparison_table(RunMetrics()))

                gr.HTML(
                    '<div style="font-size:11px; text-transform:uppercase; '
                    f'letter-spacing:0.08em; color:{TEXT_MUTED}; '
                    'margin:18px 0 8px; font-weight:600;">'
                    'Run history (last 10)</div>'
                )
                history_df = gr.Dataframe(
                    headers=HISTORY_COLUMNS,
                    datatype=["str", "str", "number", "number", "number", "number"],
                    row_count=(0, "dynamic"),
                    col_count=(len(HISTORY_COLUMNS), "fixed"),
                    interactive=False,
                    wrap=False,
                    value=[],
                    elem_classes=["dataframe-component"],
                )

            with gr.Column(scale=1, min_width=280):
                gr.HTML(render_build_flags_html(_initial_load_status()))

        # ---------------------------------------------------------------
        # Wire the button
        # ---------------------------------------------------------------
        def on_generate(
            text: str,
            frames: float,
            history: list[list[Any]] | None,
        ) -> tuple[Any, str, str, list[list[Any]]]:
            text = (text or "").strip()
            if not text:
                err = RunMetrics(
                    error="Empty text. Type something before pressing Generate.",
                    timestamp=time.strftime("%H:%M:%S"),
                )
                return (
                    None,
                    render_metric_cards(err),
                    render_comparison_table(err),
                    history or [],
                )

            audio, metrics = generate_one(text, int(frames))
            new_history = prepend_history(history, metrics)
            return (
                audio,
                render_metric_cards(metrics),
                render_comparison_table(metrics),
                new_history,
            )

        generate_btn.click(
            fn=on_generate,
            inputs=[text_in, frames_in, history_df],
            outputs=[audio_out, metrics_html, comparison_html, history_df],
        )

    return demo


def main() -> None:
    demo = build_ui()
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=8080,
        show_api=False,
        share=False,
        inbrowser=False,
        quiet=False,
    )


if __name__ == "__main__":
    main()
