"""Launch the Dash app:  /opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python scripts/run_dashboard.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.dashboard.app import create_app  # noqa: E402

if __name__ == "__main__":
    create_app().run(debug=True, host="127.0.0.1", port=8050)
