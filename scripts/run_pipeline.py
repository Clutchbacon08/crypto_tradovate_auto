from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path


def run(cmd: list[str], env: dict | None = None) -> None:
    print("\n>>", " ".join(cmd))
    p = subprocess.run(cmd, env=env, check=False)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def ensure_dirs() -> None:
    Path("data").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)


def ensure_data() -> None:
    """
    Ensures data/mbt_15m.csv exists. Uses free Binance BTCUSDT proxy data if missing.
    """
    target = Path("data/mbt_15m.csv")
    if target.exists() and target.stat().st_size > 1000:
        print(f"✅ Found historical data: {target}")
        return

    fetcher = Path("scripts/fetch_free_data.py")
    if not fetcher.exists():
        raise SystemExit("Missing scripts/fetch_free_data.py. Add it to repo to auto-fetch free data.")

    print("⬇️  data/mbt_15m.csv not found — fetching free proxy data (Binance BTCUSDT 1m → 15m)...")
    run([sys.executable, "scripts/fetch_free_data.py"])


def main() -> None:
    ensure_dirs()

    # Make sure dependencies are installed (best effort)
    req = Path("requirements.txt")
    if req.exists():
        print("📦 Installing/updating requirements...")
        run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    else:
        print("⚠️  requirements.txt not found. Skipping install step.")

    ensure_data()

    # Run 10k optimizer + exports (trials_with_folds.parquet, trades.parquet)
    print("🧠 Running optimizer (walk-forward + stability)...")
    run([sys.executable, "scripts/optimize_strategy.py"])

    # Train signal model (live veto filter)
    print("🤖 Training signal model...")
    run([sys.executable, "scripts/train_signal_model.py"])

    # Train meta model (strategy robustness learner)
    print("🧠 Training meta model...")
    run([sys.executable, "scripts/train_meta_model.py"])

    print("\n✅ PIPELINE COMPLETE")
    print("Artifacts:")
    print(" - artifacts/trials_with_folds.parquet")
    print(" - artifacts/trades.parquet")
    print("Models:")
    print(" - models/signal_model.joblib")
    print(" - models/meta_model.joblib")

    # Optional: auto-run paper mode after training
    auto_run = os.getenv("AUTO_RUN_PAPER", "false").lower() == "true"
    if auto_run:
        print("\n🚀 AUTO_RUN_PAPER=true → starting paper bot...")
        run([sys.executable, "scripts/run_paper.py"])


if __name__ == "__main__":
    main()
