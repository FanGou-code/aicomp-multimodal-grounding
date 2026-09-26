#!/usr/bin/env python3
"""Manifest-driven annotation server (local, zero API).

Serves an annotation manifest with AI boxes pre-seeded as pre-annotations.
Drag/resize adjusts a box and writes it back to the crash-safe annotation
store in real time; boxes cannot be deleted. A frame counts as annotated when
all of its objects carry a human annotation.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aicomp_grounding.annotator.server import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
