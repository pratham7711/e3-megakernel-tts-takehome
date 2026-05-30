"""Validate the three API keys in ./.env by making minimal authenticated calls
via each provider's official Python SDK.

Run from the inference-server directory:
    python3 validate_keys.py

Exits 0 if all three keys pass, 1 otherwise.

This script is deliberately careful never to print:
  - the key itself
  - the raw SDK response object (which may contain echoed key fragments)
  - any traceback that could include the Authorization header
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Load .env from the script's own directory (works regardless of CWD)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
except ImportError:
    print("[FAIL] python-dotenv not installed. "
          "Run: pip install python-dotenv --break-system-packages")
    sys.exit(1)

ENV_PATH = Path(__file__).resolve().parent / ".env"
if not ENV_PATH.exists():
    print(f"[FAIL] .env not found at {ENV_PATH}")
    sys.exit(1)
load_dotenv(ENV_PATH)


def _redact_exc(exc: BaseException) -> str:
    """Return a short error description that cannot leak the key.

    We deliberately do NOT include traceback frames or message bodies that
    could contain the Authorization header value.
    """
    name = type(exc).__name__
    # Many SDKs include status_code on their exceptions
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status:
        return f"{name} (HTTP {status})"
    return name


# ---------------------------------------------------------------------------
# 1. HuggingFace
# ---------------------------------------------------------------------------
def check_hf() -> tuple[bool, str]:
    token = os.environ.get("HF_TOKEN")
    if not token:
        return False, "HF_TOKEN missing from .env"
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return False, "huggingface_hub not installed"
    try:
        api = HfApi(token=token)
        # whoami() works across versions; whoami_v2() exists on newer ones.
        # Prefer v2, fall back to whoami().
        info = None
        if hasattr(api, "whoami_v2"):
            try:
                info = api.whoami_v2()
            except Exception:
                info = None
        if info is None:
            info = api.whoami()

        # `info` is a dict — pull only safe identity fields.
        user = info.get("name") or info.get("user") or "<unknown>"
        utype = info.get("type", "<unknown>")
        email_verified = info.get("emailVerified", info.get("email_verified", "?"))
        # Some tokens include token metadata
        auth = info.get("auth", {})
        token_role = auth.get("accessToken", {}).get("role", "?") if isinstance(auth, dict) else "?"
        return True, (f"user={user}, type={utype}, "
                      f"email_verified={email_verified}, token_role={token_role}")
    except Exception as e:  # noqa: BLE001
        return False, _redact_exc(e)


# ---------------------------------------------------------------------------
# 2. Deepgram
# ---------------------------------------------------------------------------
def check_deepgram() -> tuple[bool, str]:
    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        return False, "DEEPGRAM_API_KEY missing from .env"
    try:
        from deepgram import DeepgramClient
    except ImportError:
        return False, "deepgram-sdk not installed"
    try:
        # SDK 7.x requires kwarg; older SDKs accept positional. Try both.
        try:
            client = DeepgramClient(api_key=key)
        except TypeError:
            client = DeepgramClient(key)

        # The SDK has gone through several refactors. Try the modern path
        # first, then fall back to older paths. Each call is read-only.
        resp = None
        last_err: Exception | None = None
        for accessor in (
            lambda: client.manage.v1.projects.list(),
            lambda: client.manage.v("1").get_projects(),
            lambda: client.projects.v("1").list(),
        ):
            try:
                resp = accessor()
                break
            except (AttributeError, TypeError) as e:
                last_err = e
                continue
        if resp is None:
            raise last_err or RuntimeError("no working projects accessor on SDK")

        # Resp is either a ProjectsResponse object or a dict-like.
        projects = getattr(resp, "projects", None)
        if projects is None and isinstance(resp, dict):
            projects = resp.get("projects", [])
        projects = projects or []
        n = len(projects)
        names = []
        for p in projects[:3]:
            pname = getattr(p, "name", None)
            if pname is None and isinstance(p, dict):
                pname = p.get("name")
            names.append(pname or "<unnamed>")
        return True, f"{n} project(s), first: {names}"
    except Exception as e:  # noqa: BLE001
        return False, _redact_exc(e)


# ---------------------------------------------------------------------------
# 3. Groq
# ---------------------------------------------------------------------------
def check_groq() -> tuple[bool, str]:
    key = os.environ.get("LLM_API_KEY")
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    if provider and provider != "groq":
        return False, f"LLM_PROVIDER={provider!r} — expected 'groq'"
    if not key:
        return False, "LLM_API_KEY missing from .env"
    try:
        from groq import Groq
    except ImportError:
        return False, "groq not installed"
    try:
        client = Groq(api_key=key)
        resp = client.models.list()
        # resp.data is a list of Model objects with .id attributes
        models = getattr(resp, "data", None) or []
        ids = []
        for m in models:
            mid = getattr(m, "id", None)
            if mid is None and isinstance(m, dict):
                mid = m.get("id")
            if mid:
                ids.append(mid)
        n = len(ids)
        first3 = ids[:3]
        return True, f"{n} models available, first 3: {first3}"
    except Exception as e:  # noqa: BLE001
        return False, _redact_exc(e)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    checks = [
        ("HF_TOKEN       ", check_hf),
        ("DEEPGRAM_API_KEY", check_deepgram),
        ("LLM_API_KEY (Groq)", check_groq),
    ]
    all_ok = True
    print(f"Validating keys from {ENV_PATH}\n")
    for label, fn in checks:
        try:
            ok, msg = fn()
        except Exception as e:  # noqa: BLE001 - defensive; should not happen
            ok, msg = False, f"unhandled {_redact_exc(e)}"
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label}: {msg}")
        if not ok:
            all_ok = False
    print()
    if all_ok:
        print("All keys valid.")
        return 0
    print("One or more keys failed validation.")
    return 1


if __name__ == "__main__":
    # Hard-suppress traceback printing to stderr in case any uncaught exception
    # escapes — defensive against keys leaking via tracebacks.
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001
        print(f"[FAIL] unhandled error: {_redact_exc(e)}")
        # Discard traceback intentionally
        del traceback
        sys.exit(1)
