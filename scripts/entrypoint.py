from __future__ import annotations
import sys
from src.common.settings import settings

def main():
    mode = settings.MODE.value

    if mode == "paper":
        from scripts.run_paper import main as run
        run()
        return

    if mode == "demo":
        from scripts.run_demo import main as run
        run()
        return

    if mode == "live":
        from scripts.run_live import main as run
        run()
        return

    raise SystemExit(f"Unknown MODE={mode}. Use paper/demo/live.")

if __name__ == "__main__":
    main()
