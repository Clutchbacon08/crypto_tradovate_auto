from __future__ import annotations

import os
import sys
import time
import subprocess
from datetime import datetime

def run(cmd: list[str]) -> int:
    print("\n>>", " ".join(cmd))
    return subprocess.call(cmd)

def now_hm() -> tuple[int,int]:
    d = datetime.now()
    return d.hour, d.minute

def main():
    retrain_daily = os.getenv("RETRAIN_DAILY", "true").lower() == "true"
    retrain_hour = int(os.getenv("RETRAIN_HOUR", "2"))
    retrain_min = int(os.getenv("RETRAIN_MINUTE", "0"))

    last_retrain_date = None

    while True:
        # 1) run bot as a child process
        bot = subprocess.Popen([sys.executable, "scripts/run_live.py"])

        while True:
            time.sleep(5)

            # if bot died, restart
            code = bot.poll()
            if code is not None:
                print(f"[supervisor] bot exited code={code}, restarting...")
                break

            # 2) daily retrain
            if retrain_daily:
                h, m = now_hm()
                today = datetime.now().date()

                if h == retrain_hour and m == retrain_min and last_retrain_date != today:
                    print("[supervisor] retraining now...")
                    # stop bot
                    bot.terminate()
                    try:
                        bot.wait(timeout=30)
                    except Exception:
                        bot.kill()

                    # run pipeline (your one-command training pipeline)
                    rc = run([sys.executable, "scripts/run_pipeline.py"])
                    print("[supervisor] retrain finished rc=", rc)

                    last_retrain_date = today
                    # restart bot
                    break

        # loop restarts bot

if __name__ == "__main__":
    main()
