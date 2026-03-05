from __future__ import annotations

import os
import sys
import time
import subprocess
from datetime import datetime, date


def run(cmd: list[str]) -> int:
    print("\n>>", " ".join(cmd))
    return subprocess.call(cmd)


def now_hm() -> tuple[int, int]:
    d = datetime.now()
    return d.hour, d.minute


def select_bot_script() -> str:
    """
    Decide which bot runner to launch based on MODE.
    MODE=paper -> scripts/run_paper.py (if you have it)
    MODE=demo  -> scripts/run_demo.py
    MODE=live  -> scripts/run_live.py
    """
    mode = os.getenv("MODE", "live").lower().strip()

    if mode == "paper":
        # If you don't have run_paper.py, change this to scripts/run_bot.py or your paper runner
        return "scripts/run_paper.py"

    if mode == "demo":
        return "scripts/run_demo.py"

    if mode == "live":
        return "scripts/run_live.py"

    raise SystemExit(f"Unknown MODE={mode}. Use paper/demo/live.")


def main():
    retrain_daily = os.getenv("RETRAIN_DAILY", "true").lower() == "true"
    retrain_hour = int(os.getenv("RETRAIN_HOUR", "2"))
    retrain_min = int(os.getenv("RETRAIN_MINUTE", "0"))

    # If true, supervisor will run the training pipeline once on startup
    # (useful for fresh VPS after clone, before starting bot)
    train_on_start = os.getenv("TRAIN_ON_START", "false").lower() == "true"

    # Prevent rapid restart loops
    min_restart_seconds = int(os.getenv("MIN_RESTART_SECONDS", "10"))

    last_retrain_date: date | None = None

    # Optional: run training once at startup
    if train_on_start:
        print("[supervisor] TRAIN_ON_START=true -> running pipeline before bot start...")
        rc = run([sys.executable, "scripts/run_pipeline.py"])
        print("[supervisor] startup pipeline finished rc=", rc)

    while True:
        bot_script = select_bot_script()
        print(f"[supervisor] starting bot: {bot_script}")

        bot_start_ts = time.time()
        bot = subprocess.Popen([sys.executable, bot_script])

        while True:
            time.sleep(5)

            # If bot died, restart (with backoff)
            code = bot.poll()
            if code is not None:
                runtime = time.time() - bot_start_ts
                print(f"[supervisor] bot exited code={code} runtime={runtime:.1f}s")

                # backoff to avoid tight crash loops
                if runtime < min_restart_seconds:
                    sleep_for = min_restart_seconds - runtime
                    print(f"[supervisor] backoff {sleep_for:.1f}s before restart...")
                    time.sleep(max(0.0, sleep_for))

                print("[supervisor] restarting bot...")
                break

            # Daily retrain schedule
            if retrain_daily:
                h, m = now_hm()
                today = datetime.now().date()

                if h == retrain_hour and m == retrain_min and last_retrain_date != today:
                    print("[supervisor] retraining now...")

                    # Stop bot
                    bot.terminate()
                    try:
                        bot.wait(timeout=30)
                    except Exception:
                        bot.kill()

                    # Run pipeline (one-command training pipeline)
                    rc = run([sys.executable, "scripts/run_pipeline.py"])
                    print("[supervisor] retrain finished rc=", rc)

                    last_retrain_date = today

                    # Restart bot after retrain
                    print("[supervisor] restarting bot after retrain...")
                    break


if __name__ == "__main__":
    main()
