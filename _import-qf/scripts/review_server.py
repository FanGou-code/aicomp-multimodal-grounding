#!/usr/bin/env python3
"""Census review server — adapted for query-foundry (local, zero API).

Serves the selected frames of a census run with teacher boxes pre-seeded as
AI pre-annotations. Drag/resize adjusts a box and writes it back to the
crash-safe review store in real time; boxes cannot be deleted. A frame counts
as reviewed when all of its objects carry a human annotation.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from foundry.review.server import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
