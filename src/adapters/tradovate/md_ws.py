from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Optional, Callable

import websocket  # websocket-client

from src.adapters.tradovate.rest_client import TradovateREST


@dataclass
class Quote:
    ts: float
    last: float


class TradovateQuoteStream:
    """
    Minimal WS quote stream:
    - Connect to wss://{demo/live}.tradovateapi.com/v1/websocket
    - authorize with token
    - md/subscribeQuote for contractId
    Caches the last price + timestamp.
    """

    def __init__(self) -> None:
        self.rest = TradovateREST()
        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._last_quote: Optional[Quote] = None
        self._running = False

    @property
    def ws_url(self) -> str:
        # Tradovate WS is at /v1/websocket (demo/live). :contentReference[oaicite:3]{index=3}
        base = self.rest.base_url  # e.g., https://demo.tradovateapi.com/v1
        return base.replace("https://", "wss://") + "/websocket"

    def get_last(self) -> Optional[Quote]:
        with self._lock:
            return self._last_quote

    def _set_last(self, last: float) -> None:
        with self._lock:
            self._last_quote = Quote(ts=time.time(), last=float(last))

    def start(self) -> None:
        if self._running:
            return
        self._running = True

        contract_id = self._resolve_contract_id()
        token = self.rest.get_token().access_token

        def on_open(ws):
            # authorize
            ws.send(json.dumps({"command": "authorize", "token": token}))
            # subscribe quote
            ws.send(json.dumps({
                "command": "md/subscribeQuote",
                "contractId": contract_id
            }))

        def on_message(ws, message: str):
            try:
                data = json.loads(message)
            except Exception:
                return

            # Quote messages vary; commonly include 'last' or 'lastPrice'
            last = None
            if isinstance(data, dict):
                last = data.get("last") or data.get("lastPrice") or data.get("tradePrice")
            if last is not None:
                self._set_last(float(last))

        def on_error(ws, err):
            # ignore; supervisor restarts whole process if needed
            pass

        def on_close(ws, *args):
            pass

        self._ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )

        self._thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def _resolve_contract_id(self) -> int:
        """
        Finds contractId via REST contract/find for the configured symbol (e.g. MBT).
        """
        sym = self.rest.resolve_symbol()
        matches = self.rest.contract_find(sym)
        if not matches:
            raise RuntimeError(f"Could not resolve contractId for {sym}")
        # take first match
        cid = matches[0].get("id") or matches[0].get("contractId")
        if cid is None:
            raise RuntimeError(f"contract/find returned no id for {sym}: {matches[0]}")
        return int(cid)
