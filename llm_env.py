#!/usr/bin/env python3
"""Shared LLM / judge environment setup.

Supports:
  - yunwu.ai (OpenAI-compatible relay)
  - Azure OpenAI v1 (`https://<resource>.openai.azure.com/openai/v1/`)

Never hardcode API keys — use .env at repo root.
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

YUNWU_BASE_URL = "https://yunwu.ai/v1"
AZURE_DEFAULT_ENDPOINT = "https://openagents.openai.azure.com"
AZURE_API_VERSION = "2025-04-01-preview"
DEFAULT_MINI_MODEL = "gpt-4o-mini"

_ROOT = Path(__file__).resolve().parent

_OVERRIDE_KEYS = frozenset({
    "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_BASE_URL",
    "LLM_PROVIDER", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION",
    "STALE_MODEL", "JUDGE_MODEL", "OPENAI_MODEL", "TARGET_MODEL",
})


def azure_v1_base(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/openai/v1"):
        return endpoint + "/"
    if "/openai/" in endpoint:
        parsed = urlparse(endpoint)
        return f"{parsed.scheme}://{parsed.netloc}/openai/v1/"
    return endpoint + "/openai/v1/"


def _parse_env_file(path: Path, *, override: bool = False) -> None:
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip()
            val = v.strip().strip('"').strip("'")
            if not val or "your_api" in val.lower() or val == "...":
                continue
            if override and key in _OVERRIDE_KEYS:
                os.environ[key] = val
            else:
                os.environ.setdefault(key, val)
    except Exception:
        pass


def _is_azure_mode() -> bool:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider == "azure":
        return True
    base = os.getenv("OPENAI_API_BASE", "") + os.getenv("OPENAI_BASE_URL", "")
    return "openai.azure.com" in base or bool(os.getenv("AZURE_OPENAI_ENDPOINT"))


def _apply_azure_defaults() -> None:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", AZURE_DEFAULT_ENDPOINT).strip()
    os.environ.setdefault("AZURE_OPENAI_ENDPOINT", endpoint)
    os.environ.setdefault("AZURE_OPENAI_API_VERSION", AZURE_API_VERSION)
    base = azure_v1_base(endpoint)
    os.environ["OPENAI_API_BASE"] = base
    os.environ["OPENAI_BASE_URL"] = base
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    if azure_key:
        os.environ["OPENAI_API_KEY"] = azure_key
    else:
        os.environ.pop("OPENAI_API_KEY", None)


def _apply_yunwu_defaults() -> None:
    os.environ.setdefault("OPENAI_API_BASE", YUNWU_BASE_URL)
    os.environ.setdefault("OPENAI_BASE_URL", YUNWU_BASE_URL)


def load_llm_env(root: Path | None = None) -> dict:
    """Load .env; project root .env overrides API settings."""
    root = root or _ROOT
    stale_env = Path(r"c:\Users\Enqi Du\Documents\Downloads\STALE\STALE-main\STALE\.env")

    for env_path in (stale_env, root / "组友STALE" / ".env"):
        _parse_env_file(env_path, override=False)

    _parse_env_file(root / ".env", override=True)

    if _is_azure_mode():
        _apply_azure_defaults()
    else:
        _apply_yunwu_defaults()

    os.environ.setdefault("STALE_MODEL", DEFAULT_MINI_MODEL)
    os.environ.setdefault("JUDGE_MODEL", DEFAULT_MINI_MODEL)
    os.environ.setdefault("OPENAI_MODEL", DEFAULT_MINI_MODEL)
    os.environ.setdefault("TARGET_MODEL", DEFAULT_MINI_MODEL)

    if os.getenv("OPENAI_BASE_URL") and not os.getenv("OPENAI_API_BASE"):
        os.environ["OPENAI_API_BASE"] = os.environ["OPENAI_BASE_URL"]

    os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault("HTTP_PROXY", "")
    os.environ.setdefault("HTTPS_PROXY", "")

    return {
        "provider": "azure" if _is_azure_mode() else "yunwu",
        "api_base": os.getenv("OPENAI_API_BASE", YUNWU_BASE_URL),
        "api_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "stale_model": os.getenv("STALE_MODEL", DEFAULT_MINI_MODEL),
        "judge_model": os.getenv("JUDGE_MODEL", DEFAULT_MINI_MODEL),
    }


def judge_model() -> str:
    load_llm_env()
    return os.getenv("JUDGE_MODEL", DEFAULT_MINI_MODEL)


def stale_model() -> str:
    load_llm_env()
    return os.getenv("STALE_MODEL", DEFAULT_MINI_MODEL)
