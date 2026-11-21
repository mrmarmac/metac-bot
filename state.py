"""Encrypted GitHub Contents journal. Every authorization precedes the side effect."""
import base64
import copy
import json
import os
import threading
import uuid
import zlib
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

import requests
from cryptography.fernet import Fernet


class BudgetExceeded(RuntimeError):
    pass


class StateError(RuntimeError):
    pass


class GitHubFile:
    branch = "futureeval-state"

    def __init__(self):
        repo = os.environ["GITHUB_REPOSITORY"]
        self.url = f"https://api.github.com/repos/{repo}/contents/forecast-state.enc"
        self.cipher = Fernet(os.environ["STATE_KEY"].encode())
        self.headers = {"Authorization": "Bearer " + os.environ["GH_STATE_TOKEN"]}
        self.sha = None

    def read(self):
        response = requests.get(self.url, params={"ref": self.branch}, headers=self.headers,
                                timeout=20, allow_redirects=False)
        if response.status_code != 200:
            raise StateError(f"journal read HTTP {response.status_code}")
        data = response.json()
        self.sha = data["sha"]
        return json.loads(zlib.decompress(self.cipher.decrypt(base64.b64decode(data["content"]))))

    def write(self, data):
        compressed = zlib.compress(json.dumps(data, allow_nan=False).encode())
        content = base64.b64encode(self.cipher.encrypt(compressed)).decode()
        if len(content) > 1_000_000:
            raise StateError("journal capacity exceeded")
        body = {"message": "Update encrypted forecast journal", "branch": self.branch, "content": content}
        if self.sha:
            body["sha"] = self.sha
        response = requests.put(self.url, headers=self.headers, json=body, timeout=25, allow_redirects=False)
        if response.status_code not in (200, 201):
            raise StateError(f"journal write HTTP {response.status_code}")
        self.sha = response.json()["content"]["sha"]


def initial():
    return {"version": 1, "calls": {}, "jobs": {}}


def micros(value, rounding=ROUND_CEILING):
    amount = Decimal(str(value))
    if not amount.is_finite() or amount < 0:
        raise ValueError("expected a finite nonnegative dollar amount")
    return int((amount * 1_000_000).to_integral_value(rounding=rounding))


class Ledger:
    def __init__(self, store, per_question, monthly):
        self.store = store
        self.per_question = micros(per_question, ROUND_FLOOR)
        self.monthly = micros(monthly, ROUND_FLOOR)
        self.data = store.read()
        if self.data["version"] != 1:
            raise StateError("unknown journal version")
        self.lock = threading.RLock()
        self.healthy = True

    def change(self, operation):
        with self.lock:
            if not self.healthy:
                raise StateError("journal needs reloading")
            next_state = copy.deepcopy(self.data)
            result = operation(next_state)
            try:
                self.store.write(next_state)
            except Exception:
                self.healthy = False
                raise
            self.data = next_state
            return result

    def reserve(self, qid, max_usd):
        amount = micros(max_usd)
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        key = uuid.uuid4().hex
        def assign(data):
            calls = data["calls"].values()
            qtotal = sum(c["amount"] for c in calls if c["qid"] == qid)
            mtotal = sum(c["amount"] for c in calls if c["month"] == month)
            if data.get("provider_overrun"):
                raise BudgetExceeded("provider exceeded its price cap")
            if qtotal + amount > self.per_question or mtotal + amount > self.monthly:
                raise BudgetExceeded("forecast budget exhausted")
            data["calls"][key] = dict(amount=amount, qid=qid, month=month, settled=False)
        self.change(assign)
        return key

    def settle(self, key, actual_usd, input_tokens, output_tokens):
        amount = micros(actual_usd)
        def apply(data):
            call = data["calls"][key]
            if call["settled"]:
                raise StateError("invalid provider settlement")
            if amount > call["amount"]:
                data["provider_overrun"] = True
            call.update(amount=amount, settled=True, input_tokens=input_tokens, output_tokens=output_tokens)
            return bool(data.get("provider_overrun"))
        if self.change(apply):
            raise StateError("unexpected provider charge recorded")

    def jobs(self):
        with self.lock:
            return copy.deepcopy(self.data["jobs"])

    def record(self, qid, expected, **values):
        def update(data):
            job = data["jobs"].setdefault(str(qid), {})
            if job.get("stage") != expected:
                raise StateError("conflicting submission")
            job.update(values)
            if job["stage"] in {"done", "expired"}:
                job.pop("payload", None)
                job.pop("comment", None)
        self.change(update)

    def costs(self):
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        with self.lock:
            calls = [c for c in self.data["calls"].values() if c["month"] == month]
        return {"month_utc": month,
                "settled_usd": sum(c["amount"] for c in calls if c["settled"]) / 1e6,
                "outstanding_reserved_usd": sum(c["amount"] for c in calls if not c["settled"]) / 1e6,
                "input_tokens": sum(c.get("input_tokens", 0) or 0 for c in calls),
                "output_tokens": sum(c.get("output_tokens", 0) or 0 for c in calls)}
