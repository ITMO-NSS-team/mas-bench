from __future__ import annotations

import os

from logly import logger

# Level can be overridden via env var, e.g. BENCHLIB_LOG_LEVEL=DEBUG
_LEVEL = os.environ.get("BENCHLIB_LOG_LEVEL", "INFO").upper()

logger.configure(level=_LEVEL, color=True, console=True)

__all__ = ["logger"]
