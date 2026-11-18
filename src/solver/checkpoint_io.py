# Modified 2026-11-18: portable packaging/workflows; see CHANGELOG.md.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Replace complete checkpoint files atomically so interruptions keep the last save."""
import json
import os
from pathlib import Path
import numpy as np


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def save_fields(path, **fields):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **fields)
    os.replace(temporary, path)
