#!/usr/bin/env python3
"""Run directly from a checkout without installing any dependencies."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from clusterinfo_mc.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
