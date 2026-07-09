"""
Vercel FastAPI entrypoint (shim).

Avoids Chinese module path SRS_融合 in import name — adds backend dir to sys.path instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_BACKEND = _ROOT / "SRS_融合"
_GRAPHMEM = _ROOT / "补充内容" / "STALE_GraphMem_ATMS"

for p in (str(_ROOT), str(_BACKEND), str(_GRAPHMEM)):
    if p not in sys.path:
        sys.path.insert(0, p)

from srs_api_v13 import app  # noqa: E402

__all__ = ["app"]
