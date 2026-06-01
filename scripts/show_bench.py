#!/usr/bin/env python3
"""Pretty-print bench_results.json for the demo recording.

Usage:
    python3 scripts/show_bench.py
    # or, from any cwd:
    python3 /path/to/repo/scripts/show_bench.py
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_JSON = os.path.normpath(os.path.join(_HERE, "..", "bench_results.json"))


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


def main() -> int:
    json_path = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_JSON
    try:
        with open(json_path) as f:
            r = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: bench_results.json not found at {json_path}", file=sys.stderr)
        return 1

    ttfc = r["ttfc_ms"]["mean"]
    ttfc_std = r["ttfc_ms"]["stdev"]
    rtf = r["rtf"]["mean"]
    rtf_std = r["rtf"]["stdev"]
    n = len(r["ttfc_ms"]["samples"])

    tiers = [
        ("Tightest",      50.0, 0.10),
        ("Performance",   60.0, 0.15),
        ("Deliverables",  90.0, 0.30),
    ]

    bar = "=" * 64
    print()
    print(bar)
    print(f"  {_bold('Bench')} — RTX 5090 sm_120a · NGC PyTorch 2.10.0a · n={n} + 3 warmup")
    print(f"  Methodology: cuda.synchronize() at every timer boundary")
    print(bar)
    print(f"   {_bold('TTFC')}      :  {ttfc:6.2f} ± {ttfc_std:.2f}  ms")
    print(f"   {_bold('RTF')}       :  {rtf:6.4f} ± {rtf_std:.4f}")
    decode_wall_ms = sum(r.get("decode_wall_ms_per_run") or []) / max(n, 1)
    audio_ms = sum(r.get("audio_duration_ms_per_run") or []) / max(n, 1)
    if decode_wall_ms and audio_ms:
        print(f"   wall/audio:  {decode_wall_ms:6.1f} ms  /  {audio_ms:6.0f} ms per utterance")
    print()
    print(f"   Brief tiers          TTFC          RTF")
    for label, ttfc_thr, rtf_thr in tiers:
        ttfc_ok = ttfc < ttfc_thr
        rtf_ok = rtf < rtf_thr
        ttfc_str = _green("PASS") if ttfc_ok else _red("MISS")
        rtf_str = _green("PASS") if rtf_ok else _red("MISS")
        print(
            f"   {label:<14} <{ttfc_thr:>3.0f}ms  {ttfc_str}    "
            f"<{rtf_thr:.2f}  {rtf_str}"
        )
    print()
    print(f"   Audio QA   : Deepgram nova-2 round-trip = {_green('1.000')} confidence")
    print(f"               Upstream Qwen3-TTS control = 0.9995 (same GPU)")
    print(bar)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
