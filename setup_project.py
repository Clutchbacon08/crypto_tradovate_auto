```python
import os

project_structure = {
    ".gitignore": """
__pycache__/
*.pyc
*.log
.env
venv/
.venv/
""",

    ".env.example": """
TRADOVATE_USERNAME=
TRADOVATE_PASSWORD=
TRADOVATE_APP_ID=
TRADOVATE_APP_VERSION=1.0
TRADOVATE_ENV=demo
MODE=paper
BOT_SYMBOL=MBT
TIMEFRAME_MINUTES=15
""",

    "requirements.txt": """
pandas
numpy
scikit-learn
requests
websocket-client
python-dotenv
pydantic
""",

    "README.md": "# Crypto Tradovate Algo Bot",

    "src/common/schemas.py": """
from pydantic import BaseModel

class TradeIntent(BaseModel):
    symbol:str
    side:str
""",

    "src/common/settings.py": """
class Settings:
    SYMBOL="MBT"
""",

    "src/data/indicators.py": """
def ema(series,length):
    return series.ewm(span=length).mean()
""",

    "src/data/feature_engineering.py": """
def compute_features(df):
    return df.iloc[-1]
""",

    "src/strategy/baseline_rules.py": """
def propose_trade(features):
    return None
""",

    "src/ml/infer.py": """
def ml_filter(intent,features):
    return True
""",

    "src/risk/guardian.py": """
def evaluate_risk(intent,state):
    return True
""",

    "src/execution/executor.py": """
def execute_order(order):
    print("executing order")
""",

    "scripts/run_paper.py": '''
print("Paper trading bot running")
'''
}

for path,content in project_structure.items():

    os.makedirs(os.path.dirname(path),exist_ok=True)

    with open(path,"w") as f:
        f.write(content)

print("Project created")
```
