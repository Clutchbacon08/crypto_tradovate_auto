from __future__ import annotations

import os

from src.risk.emergency import flatten_all_positions


def main():
    sym = os.getenv("FLATTEN_SYMBOL", "").strip() or None
    res = flatten_all_positions(symbol=sym)
    print(res)


if __name__ == "__main__":
    main()
