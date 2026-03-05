from __future__ import annotations

from src.common.settings import settings

def main():
    if settings.MODE.value == "paper":
        from scripts.run_paper import main as run_paper
        run_paper()
        return

    # demo/live will be added once Tradovate adapter is wired
    raise SystemExit("MODE is demo/live but Tradovate adapter isn't wired yet.")

if __name__ == "__main__":
    main()
