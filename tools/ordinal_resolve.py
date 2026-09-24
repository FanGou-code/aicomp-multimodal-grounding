#!/usr/bin/env python3
"""CLI entry: ordinal resolve (axis ranking and the k-th pick)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aicomp_grounding.serving.ordinal.resolve import main  # noqa: E402

if __name__ == "__main__":
    main()
