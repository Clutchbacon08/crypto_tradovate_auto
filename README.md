# Crypto Tradovate Algo Bot

## VPS commands
git clone ...
cd ...
python bootstrap_repo.py
python -m venv .venv
source .venv/bin/activate (or Windows activate)
pip install -r requirements.txt
git submodule update --init --recursive

# Put data file in data/mbt_15m.csv
python scripts/pipeline_train_all.py

# Paper test with ML
python scripts/run_bot.py
