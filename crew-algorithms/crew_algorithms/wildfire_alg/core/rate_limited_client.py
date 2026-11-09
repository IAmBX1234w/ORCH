"""OpenAI-compatible clients with a shared, paced HTTP request budget.

HTTP hooks count every attempt, including SDK retries. SQLite coordinates threads,
event loops, and ORCH processes belonging to the same OS user. All models and API
keys on an endpoint share its budget; secrets are never written to the database.
"""

import asyncio
import math
import os
import sqlite3
import time
from contextlib import closing
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from openai import AsyncOpenAI as _AsyncOpenAI, OpenAI as _OpenAI


class RequestBudget:
    def __init__(self, path, scope, rpm, *, window=60.0, margin=0.05):
        if not math.isfinite(rpm) or rpm <= 0:
            raise ValueError("Request budget must be a positive finite number")
        self.path = Path(path)
        self.scope = scope
        self.interval = window / rpm + margin
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=30)) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS request_budget "
                "(scope TEXT PRIMARY KEY, next_at REAL NOT NULL)"
            )
            db.commit()

    def try_acquire(self):
        """Claim a slot now, or return how long to wait before trying again."""
        with closing(sqlite3.connect(self.path, timeout=30)) as db:
            db.execute("BEGIN IMMEDIATE")
            now = time.time()
            row = db.execute(
                "SELECT next_at FROM request_budget WHERE scope=?", (self.scope,)
            ).fetchone()
            delay = max(0.0, row[0] - now) if row else 0.0
            if delay == 0:
                db.execute(
                    "INSERT OR REPLACE INTO request_budget VALUES (?, ?)",
                    (self.scope, now + self.interval),
                )
            db.commit()
        return delay

    def defer(self, seconds):
        """Extend the shared cooldown after a provider returns HTTP 429."""
        with closing(sqlite3.connect(self.path, timeout=30)) as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "INSERT INTO request_budget VALUES (?, ?) "
                "ON CONFLICT(scope) DO UPDATE SET next_at=max(next_at, excluded.next_at)",
                (self.scope, time.time() + seconds),
            )
            db.commit()

    def before_request(self, request):
        while True:
            delay = self.try_acquire()
            if delay == 0:
                return
            time.sleep(delay)

    async def before_async_request(self, request):
        while True:
            delay = await asyncio.to_thread(self.try_acquire)
            if delay == 0:
                return
            # Waiting agents do not block the event loop or reserve future slots.
            await asyncio.sleep(delay)

    def after_response(self, response):
        if response.status_code == 429:
            delay = retry_delay(response.headers)
            self.defer(delay)
            print(f"[LLM rate limit] HTTP 429: pausing {self.scope} for {delay:.1f}s", flush=True)

    async def after_async_response(self, response):
        if response.status_code == 429:
            await asyncio.to_thread(self.after_response, response)


def retry_delay(headers):
    """Respect numeric/HTTP-date Retry-After, including delays above SDK limits."""
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        value = headers.get(name)
        if value is None:
            continue
        try:
            seconds = float(value) * scale
        except ValueError:
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                seconds = date.timestamp() - time.time()
            except (ValueError, TypeError, OverflowError):
                continue
        if math.isfinite(seconds) and seconds > 0:
            return seconds + 0.05
    return 60.0


def _install_budget(client, *, asynchronous):
    host = client.base_url.host.lower()
    is_nvidia = host == "integrate.api.nvidia.com"
    rpm = float(os.environ.get("ORCH_LLM_RPM", "40" if is_nvidia else "0"))
    if not math.isfinite(rpm) or rpm < 0 or (is_nvidia and rpm == 0):
        raise ValueError("ORCH_LLM_RPM must be positive for NVIDIA (0 disables other endpoints)")
    if is_nvidia:
        rpm = min(rpm, 40.0)
    if rpm == 0:
        return

    # Reused clients (including SDK with_options copies) must not acquire twice.
    hooks = client._client.event_hooks
    if any(isinstance(getattr(hook, "__self__", None), RequestBudget)
           for hook in hooks.get("request", [])):
        return
    path = os.environ.get(
        "ORCH_RATE_LIMIT_DB", str(Path.home() / ".cache" / "orch" / "request_budget.sqlite3")
    )
    budget = RequestBudget(path, host, rpm)
    hooks.setdefault("request", []).append(
        budget.before_async_request if asynchronous else budget.before_request
    )
    hooks.setdefault("response", []).append(
        budget.after_async_response if asynchronous else budget.after_response
    )


class OpenAI(_OpenAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _install_budget(self, asynchronous=False)


class AsyncOpenAI(_AsyncOpenAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _install_budget(self, asynchronous=True)
