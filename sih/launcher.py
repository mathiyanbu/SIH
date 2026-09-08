"""One-click launcher for the AI noise cancellation dashboard."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_PATH = ROOT / "main.py"


def main():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    flags = 0
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    subprocess.Popen([sys.executable, str(APP_PATH)], cwd=str(ROOT), env=env, creationflags=flags)


if __name__ == "__main__":
    main()
