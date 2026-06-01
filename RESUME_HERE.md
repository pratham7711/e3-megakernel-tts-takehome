# Resume marker (personal session diary)

> This file is the author's internal workflow notes — useful when picking up the project between sessions. **It is not a submission deliverable.**
>
> For the canonical view of the project, see:
>
> - **[`README.md`](./README.md)** — architecture, kernel mods, how to run (the brief's "Short README")
> - **[`ENGINEERING_NOTES.md`](./ENGINEERING_NOTES.md)** — process, tradeoffs, results, why we miss Tightest tier
> - **[`CHANGELOG.md`](./CHANGELOG.md)** — chronological diff with bench numbers
> - **[`DEMO_SCRIPT.md`](./DEMO_SCRIPT.md)** — recording script for the demo video
> - **[`bench_results.json`](./bench_results.json)** — canonical numbers (TTFC 25.32 ± 0.03 ms · RTF 0.1452 ± 1.7e-4)

## Current state (2026-06-01)

- Audio: Deepgram round-trip **1.000** confidence on bot output (upstream control: 0.9995). Matches.
- Perf: passes **Performance tier** (TTFC < 60 / RTF < 0.15) on both metrics. Misses Tightest tier on RTF by 0.045.
- Megakernel-AR swap landed: `torch.ops.qwen_megakernel_C.decode_embed` is the production AR talker step. Closed RTF 0.181 → 0.145.
- Repo + demo + email all aligned to the same numbers.

## Open next moves (not in this submission)

1. CP mega-graph — collapse 14 per-step CUDA-graph replays into one. Est. 3-5 ms/step, projects RTF ~0.10-0.11. ~60-75 min.
2. Pipeline-by-one — issue next step before consuming current codec frame. Est. 2-3 ms/step on top. Risks TTFC regression; needs first-yield guard. ~30-45 min.

Both detailed in `CHANGELOG.md`'s "next move" footer.
