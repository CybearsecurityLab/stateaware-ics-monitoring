"""Test-session bootstrap for the repository's ``src`` layout.

The project is installed in editable mode, but the editable install only
exposes the packages listed in ``[tool.hatch.build.targets.wheel]``.  Adding
the source tree to ``sys.path`` makes every working-tree package importable,
so a test can exercise code that is not (yet) part of the wheel manifest.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
