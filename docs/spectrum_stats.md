# Real-codec audio — quick stats

Source: `bench_audio.wav` (24 kHz mono, ~2 s).

| metric | value |
|---|---|
| peak amplitude | 0.2500 |
| RMS | 0.1728 |
| spectral centroid | 976.6 Hz |
| dynamic range | 23.9 dB |
| duration | 2.000 s |
| sample rate | 24000 Hz |

Spectrogram is multi-component / broadband (energy spread across ~200-2000 Hz with
formant-like structure), not a single sine tone — confirming the Qwen3-TTS real-codec
decode path actually runs end-to-end.
