"""No-GPU voice-loop validation UI.

Browser-based UI (Gradio) that exercises the SAME interaction the real demo
will: speak into the mic, get audio back through the speakers. Uses the three
external APIs we'll need in production (Deepgram STT, Groq LLM, HF for
weights) and substitutes macOS ``say`` for the megakernel TTS so the loop can
run on a laptop with no GPU.

Why this exists:
    Validates the **UI / transport / API** layer of the brief's Step 3 pipeline
    independently of the megakernel work. If this loop works end-to-end
    locally, we know mic + speaker permissions, Gradio audio plumbing, and
    every external API key are good. Then the only thing left to verify on
    GPU is the megakernel TTS substitution for ``say``.

Two tabs:
    1. Loopback — just echo the mic input back with a 1 s leading silence.
       Tests that the BROWSER captures audio and the BROWSER plays it back.
    2. Full pipeline — mic → Deepgram STT → Groq LLM → macOS ``say`` →
       browser playback. Tests every API + every plumbing seam.

Run:
    cd inference-server && python3 ui_loopback.py
    # opens http://127.0.0.1:7861 in your default browser
"""
from __future__ import annotations

import io
import os
import subprocess
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

# .env lives next to this script.
load_dotenv(Path(__file__).parent / ".env")

DEEPGRAM_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")


# ---------------------------------------------------------------------------
# External API calls
# ---------------------------------------------------------------------------


def call_deepgram_stt(wav_bytes: bytes) -> tuple[str | None, str | None]:
    """Transcribe a WAV via Deepgram REST. Returns (transcript, error_or_None)."""
    if not DEEPGRAM_KEY:
        return None, "DEEPGRAM_API_KEY missing from .env"
    import requests

    r = requests.post(
        "https://api.deepgram.com/v1/listen?model=nova-2&language=en-US&punctuate=true",
        headers={
            "Authorization": f"Token {DEEPGRAM_KEY}",
            "Content-Type": "audio/wav",
        },
        data=wav_bytes,
        timeout=30,
    )
    if r.status_code != 200:
        return None, f"Deepgram HTTP {r.status_code}: {r.text[:200]}"
    try:
        text = (
            r.json()["results"]["channels"][0]["alternatives"][0]["transcript"]
        )
    except Exception as e:  # noqa: BLE001
        return None, f"Deepgram parse error: {e!r}"
    return text, None


def call_groq_llm(text: str) -> tuple[str | None, str | None]:
    """Get a 1-2 sentence assistant reply from Groq."""
    if not LLM_API_KEY:
        return None, "LLM_API_KEY missing from .env"
    try:
        from groq import Groq
    except Exception as e:  # noqa: BLE001
        return None, f"groq import failed: {e!r}"
    client = Groq(api_key=LLM_API_KEY)
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "llama-3.1-8b-instant"),
            messages=[
                {"role": "system", "content": "You are a friendly voice agent. Reply in 1-2 short sentences."},
                {"role": "user", "content": text},
            ],
            temperature=0.7,
            max_tokens=200,
        )
    except Exception as e:  # noqa: BLE001
        return None, f"Groq API call failed: {e!r}"
    return resp.choices[0].message.content, None


def call_hf_whoami() -> tuple[str | None, str | None]:
    """Validate the HF token by hitting whoami."""
    if not HF_TOKEN:
        return None, "HF_TOKEN missing from .env"
    import requests

    r = requests.get(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        timeout=10,
    )
    if r.status_code != 200:
        return None, f"HF HTTP {r.status_code}: {r.text[:200]}"
    try:
        j = r.json()
        return j.get("name", "<unknown>"), None
    except Exception as e:  # noqa: BLE001
        return None, f"HF parse error: {e!r}"


def mac_say_tts(text: str) -> str | None:
    """Use macOS ``say`` to synthesise ``text`` to a 24 kHz mono int16 WAV.

    Returns the WAV path, or ``None`` if the local tools are not available.
    """
    if not text or not text.strip():
        return None
    try:
        aiff_fd, aiff_path = tempfile.mkstemp(suffix=".aiff")
        os.close(aiff_fd)
        wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(wav_fd)
        subprocess.run(["say", "-o", aiff_path, text], check=True, timeout=30)
        subprocess.run(
            [
                "afconvert",
                "-f", "WAVE",
                "-d", "LEI16@24000",
                "-c", "1",
                aiff_path,
                wav_path,
            ],
            check=True,
            timeout=20,
        )
        os.unlink(aiff_path)
        return wav_path
    except Exception as e:  # noqa: BLE001
        print(f"[mac_say_tts] error: {e!r}")
        return None


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------


def loopback_only(audio_input):
    """Echo the mic input back with 1 s leading silence."""
    if audio_input is None:
        return None, "No audio captured. Click the record button and speak."
    sr, data = audio_input
    if data is None or len(data) == 0:
        return None, "Captured 0 samples."
    if data.dtype != np.int16:
        # Gradio sometimes hands back float32 in [-1, 1].
        data = np.clip(data, -1.0, 1.0)
        data = (data * 32767.0).astype(np.int16)
    delay = np.zeros(sr, dtype=np.int16)
    out = np.concatenate([delay, data])
    return (
        (sr, out),
        f"Loopback OK: captured {len(data)} samples @ {sr} Hz "
        f"({len(data)/sr:.2f} s). Output has 1 s leading silence then your voice.",
    )


def full_pipeline(audio_input):
    """mic → Deepgram STT → Groq LLM → macOS say → browser playback."""
    if audio_input is None:
        return None, "No audio captured. Click the record button and speak first."
    sr, data = audio_input
    if data is None or len(data) == 0:
        return None, "Captured 0 samples."
    if data.dtype != np.int16:
        data = np.clip(data, -1.0, 1.0)
        data = (data * 32767.0).astype(np.int16)

    # Write the mic input to a WAV in memory so we can POST it.
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    log_lines = []

    # ---- STT ----
    t0 = time.perf_counter()
    text, err = call_deepgram_stt(wav_bytes)
    stt_ms = (time.perf_counter() - t0) * 1000
    if err:
        return None, f"Deepgram STT failed:\n{err}"
    if not text or not text.strip():
        return None, f"STT returned empty transcript (mic sr={sr}, samples={len(data)}). Try speaking louder / closer."
    log_lines.append(f"[STT {stt_ms:.0f} ms]  You said: {text.strip()!r}")

    # ---- LLM ----
    t0 = time.perf_counter()
    reply, err = call_groq_llm(text.strip())
    llm_ms = (time.perf_counter() - t0) * 1000
    if err:
        return None, "\n".join(log_lines + [f"Groq LLM failed: {err}"])
    log_lines.append(f"[LLM {llm_ms:.0f} ms]  Assistant: {reply.strip()!r}")

    # ---- TTS (macOS say -- stand-in for megakernel TTS) ----
    t0 = time.perf_counter()
    wav_path = mac_say_tts(reply.strip())
    tts_ms = (time.perf_counter() - t0) * 1000
    if wav_path is None:
        return None, "\n".join(log_lines + [f"macOS say synthesis failed."])
    log_lines.append(f"[TTS {tts_ms:.0f} ms]  macOS `say` -> {Path(wav_path).name}")

    audio, sr_out = sf.read(wav_path)
    if audio.dtype != np.int16:
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767.0).astype(np.int16)
    try:
        os.unlink(wav_path)
    except OSError:
        pass

    total_ms = stt_ms + llm_ms + tts_ms
    log_lines.append(f"[TOTAL {total_ms:.0f} ms]  mic → speaker round trip (no GPU)")

    return (sr_out, audio), "\n".join(log_lines)


def validate_apis():
    """One-shot health check for all 3 external APIs."""
    rows = []
    # Deepgram
    if not DEEPGRAM_KEY:
        rows.append(["Deepgram", "MISSING", "DEEPGRAM_API_KEY not in .env"])
    else:
        # Dummy silent 0.1 s WAV
        silent = np.zeros(2400, dtype=np.int16)
        buf = io.BytesIO()
        sf.write(buf, silent, 24000, format="WAV", subtype="PCM_16")
        text, err = call_deepgram_stt(buf.getvalue())
        if err:
            rows.append(["Deepgram", "FAIL", err[:80]])
        else:
            rows.append(["Deepgram", "OK", f"empty transcript on silence (expected): {text!r}"])
    # Groq
    if not LLM_API_KEY:
        rows.append(["Groq", "MISSING", "LLM_API_KEY not in .env"])
    else:
        reply, err = call_groq_llm("Say only the word: hello")
        if err:
            rows.append(["Groq", "FAIL", err[:80]])
        else:
            rows.append(["Groq", "OK", f"reply: {reply!r}"])
    # HF
    name, err = call_hf_whoami()
    if err:
        rows.append(["HuggingFace", "FAIL", err[:80]])
    else:
        rows.append(["HuggingFace", "OK", f"user: {name}"])

    return rows


# ---------------------------------------------------------------------------
# Gradio Blocks
# ---------------------------------------------------------------------------


with gr.Blocks(title="e3 — Voice Loop (No-GPU Test)") as demo:
    gr.Markdown("# e3 — Voice Loop, No-GPU Validation")
    gr.Markdown(
        "**What this tests:** mic capture in browser, speaker playback in browser, "
        "and the three external APIs (Deepgram, Groq, HF) we'll use in the real "
        "pipeline. macOS `say` substitutes for the megakernel TTS so we can run "
        "this whole loop on a laptop. Once this works, the GPU restart only has "
        "to swap `say` for the real Qwen3-TTS pipeline."
    )

    with gr.Tab("1. Loopback (no APIs, just mic + speaker)"):
        gr.Markdown(
            "**Purpose:** prove that the browser asks for mic permission, captures "
            "audio, and plays it back. If this fails, none of the rest will work."
        )
        with gr.Row():
            mic1 = gr.Audio(sources=["microphone"], type="numpy", label="Speak (click mic icon → record → stop)")
        with gr.Row():
            btn1 = gr.Button("Echo my voice back", variant="primary")
        out1 = gr.Audio(label="Playback (1 s silence + your voice)", autoplay=True)
        log1 = gr.Textbox(label="Status", interactive=False)
        btn1.click(loopback_only, inputs=mic1, outputs=[out1, log1])

    with gr.Tab("2. Full pipeline (mic → Deepgram → Groq → say → speaker)"):
        gr.Markdown(
            "**Purpose:** prove all 3 APIs work end-to-end. Speak a question. "
            "Deepgram transcribes, Groq replies, macOS `say` speaks the reply back."
        )
        with gr.Row():
            mic2 = gr.Audio(sources=["microphone"], type="numpy", label="Speak (click mic → record → stop)")
        with gr.Row():
            btn2 = gr.Button("Run pipeline", variant="primary")
        out2 = gr.Audio(label="Assistant reply (via macOS say)", autoplay=True)
        log2 = gr.Textbox(label="Per-stage log", interactive=False, lines=6)
        btn2.click(full_pipeline, inputs=mic2, outputs=[out2, log2])

    with gr.Tab("3. API health check (no audio)"):
        gr.Markdown("One-shot probe of Deepgram, Groq, and HF. No mic interaction.")
        btn3 = gr.Button("Run probes", variant="primary")
        out3 = gr.Dataframe(
            headers=["API", "Status", "Detail"],
            datatype=["str", "str", "str"],
            interactive=False,
        )
        btn3.click(validate_apis, inputs=None, outputs=out3)


if __name__ == "__main__":
    print("Starting Gradio at http://127.0.0.1:7861 — opening in browser...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=True,
        share=False,
        show_error=True,
    )
