# Demo Script — e3 Take-Home

**Target: 3 minutes, conversational.** Like telling the story to a smart friend over coffee — not presenting slides.
Screen: Browser at `http://localhost:8080` on the main display, a terminal in a corner ready to show `bench_results.json`.

---

## 1. Open (20s)

> "Hey — I'm Pratham. The take-home: there's this CUDA kernel from a researcher named AlpinDale, runs a Qwen language model at a thousand tokens a second on a single 5090. The brief was, wire it up to drive a text-to-speech model, then plug that into a voice-agent pipeline. Speech in, AI reply, speech out — all on one GPU.
>
> Let me just show you it working first, then I want to tell you the most interesting thing that happened along the way."

## 2. Show it (30s)

*[Open Chrome at localhost:8080. Click the record button. Speak naturally:]*

> "Hey, what's the weather like in Mumbai tomorrow?"

*[Click Send. Watch the metric cards + stage timings fill in. Bot audio plays through the speakers — clear English reply.]*

> "So that all happened on the GPU. Voice in, voice out. The audio you heard came out streaming — chunk by chunk while it was still being generated. That's what the brief calls out as important: don't buffer the whole utterance before sending. No waiting for the full sentence."

## 3. The story worth telling — the bug that almost shipped (90s)

> "OK so here's the part I actually want to tell you about.
>
> The brief gives three difficulty tiers for performance. I'll call them tightest, middle, and easy. My first pass beat the tightest tier on both speed metrics. Eighteen milliseconds time-to-first-audio, point one two real-time factor. Beautiful numbers. I was about ten minutes from packaging it up and emailing it in.
>
> And then a voice in my head said — just check one more thing. So I took the speech the system was producing and fed it back into Deepgram, which is the speech-to-text model on the input side. Like — let me check my own work. The model that listens, can it understand what the model that talks just said?
>
> Zero confidence. Every test. The output was speech-shaped. It sounded like a person speaking from another room through a wall. But when you really listened, there were no actual words in it. Just acoustic energy in roughly the right frequency band.
>
> So I dug in. There were four bugs stacked on top of each other. Each one looked harmless on its own — like, you'd write the code that way and it would feel right. But all four together broke the voice completely.
>
> The simplest one to describe: this model is trained to pick its next sound by sampling — weighted random choice across multiple options. I was just picking the highest-probability one every time. Sounds smarter, right? Always pick the best? Turns out it makes the model babble — it falls into loops of similar sounds.
>
> The other three are similar — places where the model expects something a certain way, and we were giving it something slightly different. Each one alone would have been recoverable. All four together — the audio looked right on a spectrogram, sounded vaguely like English, but had no actual phonemes in it.
>
> I fixed all four. Audio came alive. I re-ran the speech-to-text check — perfect confidence now, matches what the original model produces.
>
> The catch: doing it correctly is heavier. The benchmarks regressed. I dropped from the tightest tier to missing the middle tier on real-time factor."

## 4. Today's fix — closing the gap (40s)

> "Then earlier today I noticed something else. The whole project is called megakernel-tts. We talk about the megakernel everywhere — the README, the file names. But when I actually traced the code path that runs during a voice turn, the megakernel wasn't being called. It was alive only in the bench harnesses. The real speech generation had quietly switched over to a slower regular-PyTorch path months ago, because the megakernel's expected input didn't quite match what the speech model needed.
>
> So I went into the CUDA kernel itself — about eighty lines of changes — and added a new input mode that accepts the shape we actually need. Wired the speech path through it.
>
> Speed jumped up. We're now twenty percent faster than this morning. Audio quality stayed perfect — speech-to-text round-trip still at one point zero confidence. We're back to passing the middle performance tier on both metrics."

## 5. The numbers (20s)

*[Switch to terminal. Run live:]*
```
python3 scripts/show_bench.py
```
*(Prints a colored table — green PASS / red MISS against each brief tier. Two cells red on the Tightest row, everything else green. Tells the whole story in one screen.)*

> "Time-to-first-audio: twenty-five milliseconds. The brief asked for under sixty. Real-time factor: point one four five. The brief asked for under point one five.
>
> Audio quality: one point zero zero zero on Deepgram round-trip. Matches the original Qwen model line for line.
>
> The only target I'm missing is the tightest one — needs the real-time factor under point one. I'm at point one four five. The gap is real but mechanical — there's one more optimization scoped out in the README, would take another hour or so. Just not in this session."

## 6. Close (15s)

> "Repo is public — link's in the email. README has everything — the four-bug story in detail, the methodology, the kernel diff. Receipts and email coming separately.
>
> Thanks for taking the time. Happy to walk through any part of it whenever."

---

## Recording notes

- **Tone**: smart friend over coffee. Pause between thoughts. The bug-discovery story is the highest-value 90 seconds — don't rush it.
- **Browser**: refresh before you record so the metric cards start blank, then watch them populate when you click Send.
- **Terminal**: pre-cd into the repo, have the `jq` command in your scrollback ready to up-arrow-Enter.
- **Failsafe**: if the browser glitches, switch to `bash scripts/run_voice_turn.sh samples/user_utterance.wav /tmp/bot_demo.wav` — file-mode round-trip, plays the bot WAV through `afplay`. Same backend, no UI.
- **Length cap**: 3 minutes target, 5 minutes hard ceiling. Section 3 (the diagnosis story) is the part nobody else will have. Tell it like a story.
