from __future__ import annotations

import time
import requests
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.common.settings import settings


@dataclass
class TradovateToken:
    access_token: str
    expiration_time: int  # epoch ms per Tradovate response (commonly)

    def is_expired(self, skew_seconds: int = 30) -> bool:
        # Tradovate example returns expirationTime; treat it as ms epoch if large
        exp_ms = int(self.expiration_time)
        now_ms = int(time.time() * 1000)
        return now_ms >= (exp_ms - skew_seconds * 1000)


class TradovateREST:
    """
    Option A implementation:
    - REST auth
    - REST orders
    - REST polling for account/positions
    """

    def __init__(self) -> None:
        self._token: Optional[TradovateToken] = None

    @property
    def base_url(self) -> str:
        env = (settings.TRADOVATE_ENV or "demo").lower()
        return settings.TRADOVATE_BASE_LIVE if env == "live" else settings.TRADOVATE_BASE_DEMO

    def _headers(self) -> Dict[str, str]:
        tok = self.get_token().access_token
        return {
            "Authorization": f"Bearer {tok}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def get_token(self) -> TradovateToken:
        if self._token is not None and not self._token.is_expired():
            return self._token

        url = f"{self.base_url}/auth/accesstokenrequest"
        payload = {
            "name": settings.TRADOVATE_USERNAME,
            "password": settings.TRADOVATE_PASSWORD,
            "appId": settings.TRADOVATE_APP_ID,
            "appVersion": settings.TRADOVATE_APP_VERSION,
            "cid": settings.TRADOVATE_CID,
            "deviceId": settings.TRADOVATE_DEVICE_ID,
            "sec": settings.TRADOVATE_API_SECRET,
        }

        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        jd = r.json()
        self._token = TradovateToken(
            access_token=jd["accessToken"],
            expiration_time=int(jd["expirationTime"]),
        )
        return self._token

    def account_list(self) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/account/list"
        r = requests.get(url, headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json()

    def resolve_account_id(self) -> int:
        if settings.TRADOVATE_ACCOUNT_ID:
            return int(settings.TRADOVATE_ACCOUNT_ID)

        accounts = self.account_list()
        if not accounts:
            raise RuntimeError("No Tradovate accounts returned from /account/list")
        return int(accounts[0]["id"])

    def contract_find(self, name: str) -> List[Dict[str, Any]]:
        # Tradovate examples commonly use /contract/find?name=...
        url = f"{self.base_url}/contract/find"
        r = requests.get(url, headers=self._headers(), params={"name": name}, timeout=30)
        r.raise_for_status()
        return r.json()

    def resolve_symbol(self) -> str:
        """
        Futures symbols are often contract-specific. If user sets TRADOVATE_SYMBOL to exact
        contract name, we use it. Otherwise we attempt contract/find and take first.
        """
        sym = settings.TRADOVATE_SYMBOL.strip()
        if len(sym) >= 4:  # may already be a full contract symbol
            return sym

        # Try to find best match for "MBT"
        matches = self.contract_find(sym)
        if not matches:
            return sym  # fall back; may still work if broker resolves
        # Heuristic: first result
        return matches[0].get("name") or sym

    def positions(self, account_id: int) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/position/list"
        r = requests.get(url, headers=self._headers(), params={"accountId": account_id}, timeout=30)
        r.raise_for_status()
        return r.json()

    def place_market_order(self, account_id: int, symbol: str, action: str, qty: int) -> Dict[str, Any]:
        """
        action: "Buy" or "Sell"
        """
        url = f"{self.base_url}/order/placeorder"
        payload = {
            "accountId": int(account_id),
            "action": action,
            "symbol": symbol,
            "orderQty": int(qty),
            "orderType": "Market",
            "isAutomated": True,
        }
        r = requests.post(url, headers=self._headers(), json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def cancel_all_working(self, account_id: int) -> None:
        # Optional convenience; endpoint differs by API versions, so this is a placeholder.
        # We keep it here so you can add later without changing bot structure.
        return
