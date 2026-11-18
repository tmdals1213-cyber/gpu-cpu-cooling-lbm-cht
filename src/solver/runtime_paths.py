# Modified 2026-11-18: portable packaging/workflows; see CHANGELOG.md.
"""Portable paths. Generated fields and caches are never stored alongside code."""
from pathlib import Path
import os
REPOSITORY=Path(__file__).resolve().parents[2]
ASSETS=REPOSITORY/'assets'
OUTPUT=Path(os.environ.get('COOLING_OUTPUT_ROOT',str(REPOSITORY/'runs/default'))).expanduser().resolve()
CACHE=Path(os.environ.get('COOLING_CACHE_ROOT',str(REPOSITORY/'.cache'))).expanduser().resolve()
OUTPUT.mkdir(parents=True,exist_ok=True)
