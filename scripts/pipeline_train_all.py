from __future__ import annotations

import os
import subprocess
import sys

def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd))
    p = subprocess.run(cmd, check=False)
    if p.returncode != 0:
        raise SystemExit(p.returncode)

def main():
    # Ensure folders exist
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 1) 10k optimization + export datasets for ML
    run([sys.executable, "scripts/optimize_strategy.py"])

    # 2) Train models
    run([sys.executable, "scripts/train_signal_model.py"])
    run([sys.executable, "scripts/train_meta_model.py"])

    print("\n✅ Training pipeline complete.")
    print("Artifacts:")
    print(" - artifacts/trials_with_folds.parquet")
    print(" - artifacts/trades.parquet")
    print("Models:")
    print(" - models/signal_model.joblib")
    print(" - models/meta_model.joblib")

if __name__ == "__main__":
    main()
