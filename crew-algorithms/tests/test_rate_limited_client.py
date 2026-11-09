import asyncio
import ast
import multiprocessing
import os
import tempfile
import time
import unittest
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from unittest.mock import patch

import httpx

from crew_algorithms.wildfire_alg.core.rate_limited_client import (
    AsyncOpenAI, OpenAI, RequestBudget, retry_delay,
)


def acquire_in_process(path, ready, output):
    budget = RequestBudget(path, "shared", 1, window=0.04, margin=0.01)
    ready.wait()
    budget.before_request(None)
    output.put(time.time())


def completion():
    return {"id": "test", "object": "chat.completion", "created": 0,
            "model": "test", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = str(Path(self.temp.name) / "budget.sqlite3")

    def test_rolling_minute_and_no_idle_burst(self):
        budget = RequestBudget(self.path, "nvidia", 40)
        sends = []
        now = 1000.0
        with patch("crew_algorithms.wildfire_alg.core.rate_limited_client.time.time") as clock:
            for _ in range(100):
                clock.return_value = now
                delay = budget.try_acquire()
                if delay:
                    now += delay + 0.000001
                    clock.return_value = now
                    self.assertEqual(budget.try_acquire(), 0)
                sends.append(now)
            for start in sends:
                self.assertLessEqual(sum(start <= sent <= start + 60 for sent in sends), 40)
            clock.return_value = now + 600
            self.assertEqual(budget.try_acquire(), 0)
            self.assertGreater(budget.try_acquire(), 1.5)

    def test_multiple_processes_share_pacing(self):
        context = multiprocessing.get_context("spawn")
        ready, output = context.Event(), context.Queue()
        processes = [context.Process(target=acquire_in_process,
                     args=(self.path, ready, output)) for _ in range(4)]
        try:
            for process in processes:
                process.start()
            ready.set()
            sent = sorted(output.get(timeout=30) for _ in processes)
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)
            self.assertTrue(all(b - a >= 0.04 for a, b in zip(sent, sent[1:])))
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join()
            output.close()

    def test_cooldown_shared_and_never_shortened(self):
        one = RequestBudget(self.path, "same", 40)
        two = RequestBudget(self.path, "same", 40)
        with patch("crew_algorithms.wildfire_alg.core.rate_limited_client.time.time", return_value=1000):
            one.defer(120)
            two.defer(10)
            self.assertEqual(two.try_acquire(), 120)

    def test_retry_after_formats(self):
        self.assertAlmostEqual(retry_delay({"retry-after": "120"}), 120.05)
        self.assertAlmostEqual(retry_delay({"retry-after-ms": "2500"}), 2.55)
        with patch("crew_algorithms.wildfire_alg.core.rate_limited_client.time.time", return_value=1000):
            date = format_datetime(datetime.fromtimestamp(1090, timezone.utc), usegmt=True)
            self.assertAlmostEqual(retry_delay({"retry-after": date}), 90.05)
        for value in ("invalid", "nan", "-1", "0"):
            self.assertEqual(retry_delay({"retry-after": value}), 60)


class ClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = str(Path(self.temp.name) / "budget.sqlite3")
        env = patch.dict(os.environ, {"ORCH_RATE_LIMIT_DB": self.path, "ORCH_LLM_RPM": "40"})
        env.start()
        self.addCleanup(env.stop)

    async def test_concurrent_clients_and_sync_thread_share_budget(self):
        sends = []

        def handler(request):
            sends.append(time.monotonic())
            return httpx.Response(200, json=completion())

        kwargs = {"api_key": "fake", "base_url": "https://integrate.api.nvidia.com/v1"}
        sync = OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        first = AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        second = AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        call = {"model": "test", "messages": [{"role": "user", "content": "test"}]}
        try:
            responses = await asyncio.gather(
                first.chat.completions.create(**call),
                second.chat.completions.create(**call),
                asyncio.to_thread(sync.chat.completions.create, **call),
            )
            self.assertTrue(all(r.choices[0].message.content == "ok" for r in responses))
            sends.sort()
            self.assertTrue(all(b - a >= 1.5 for a, b in zip(sends, sends[1:])))
        finally:
            sync.close()
            await first.close()
            await second.close()

    async def test_sdk_retry_passes_through_budget(self):
        sends = []

        def handler(request):
            sends.append(time.monotonic())
            if len(sends) == 1:
                return httpx.Response(429, headers={"retry-after": "0.01"},
                                      json={"error": {"message": "slow down"}})
            return httpx.Response(200, json=completion())

        async with AsyncOpenAI(api_key="fake", base_url="https://integrate.api.nvidia.com/v1",
                               http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))) as client:
            result = await client.chat.completions.create(model="test", messages=[])
            self.assertEqual(result.choices[0].message.content, "ok")
        self.assertEqual(len(sends), 2)
        self.assertGreaterEqual(sends[1] - sends[0], 1.5)

    async def test_cancelled_wait_does_not_send_request(self):
        calls = []
        budget = RequestBudget(self.path, "integrate.api.nvidia.com", 40)
        budget.defer(60)
        async with AsyncOpenAI(api_key="fake", base_url="https://integrate.api.nvidia.com/v1",
                               http_client=httpx.AsyncClient(transport=httpx.MockTransport(
                                   lambda request: calls.append(request)))) as client:
            task = asyncio.create_task(client.chat.completions.create(model="test", messages=[]))
            await asyncio.sleep(0.05)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(calls, [])

    async def test_nvidia_cap_and_sdk_client_copy(self):
        with patch.dict(os.environ, {"ORCH_LLM_RPM": "100"}):
            async with AsyncOpenAI(api_key="fake", base_url="https://integrate.api.nvidia.com/v1") as client:
                budget = client._client.event_hooks["request"][0].__self__
                self.assertEqual(budget.interval, 1.55)
                copied = client.with_options(timeout=5)
                self.assertEqual(len(copied._client.event_hooks["request"]), 1)

    async def test_other_endpoint_can_disable_budget(self):
        with patch.dict(os.environ, {"ORCH_LLM_RPM": "0"}):
            async with AsyncOpenAI(api_key="fake", base_url="http://localhost:8000/v1") as client:
                self.assertEqual(client._client.event_hooks["request"], [])

    async def test_all_orch_clients_use_limiter(self):
        root = Path(__file__).resolve().parents[1] / "crew_algorithms/wildfire_alg/algorithms/ORCH"
        for file in root.glob("*.py"):
            for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and node.module == "openai":
                    self.assertFalse(any(alias.name in {"OpenAI", "AsyncOpenAI"}
                                         for alias in node.names), str(file))


if __name__ == "__main__":
    unittest.main()
