"""Autonomous FutureEval entry point. Only operational counters leave this process."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import requests

from formats import FIELDS, Question, validate_payload
from state import GitHubFile, Ledger, StateError, initial


class ApiFailure(RuntimeError):
    pass


class RejectedForRateLimit(ApiFailure):
    pass


def now():
    return datetime.now(timezone.utc)


class Metaculus:
    base = "https://www.metaculus.com/api"

    def __init__(self, mode="test-area"):
        self.mode = mode
        self.headers = {"Authorization": "Token " + os.environ["METACULUS_TOKEN"]}
        self.author = int(os.environ["METACULUS_BOT_ID"])

    def request(self, method, path, **kwargs):
        # Mutation delivery may be ambiguous. Never let an HTTP adapter replay it.
        attempts = 3 if method == "GET" else 1
        for index in range(attempts):
            delay = 2 ** index
            try:
                response = requests.request(method, self.base + path, headers=self.headers,
                                            timeout=(4, 10), allow_redirects=False, **kwargs)
                if 200 <= response.status_code < 300:
                    return response.json() if method == "GET" else None
                if method == "POST" and response.status_code == 429:
                    raise RejectedForRateLimit("rate limited")
                if response.status_code not in {429, 502, 503, 504}:
                    raise ApiFailure(f"HTTP {response.status_code}")
                retry = response.headers.get("Retry-After", "")
                if retry:
                    if not retry.isdigit() or int(retry) > 20:
                        raise ApiFailure("rate limited")
                    delay = max(delay, int(retry))
            except (requests.Timeout, requests.ConnectionError):
                pass
            if index + 1 < attempts:
                time.sleep(delay)
        raise ApiFailure("remote request incomplete")

    def verify_account(self):
        try:
            profile = self.request("GET", "/users/me/")
        except ApiFailure as exc:
            # Restricted API gateway may expose only the own-ID profile path.
            if str(exc) != "HTTP 403":
                raise
            profile = self.request("GET", f"/users/{self.author}/")
        if profile["id"] != self.author or profile["is_bot"] is not True:
            raise ApiFailure("expected a bot profile")

    def pages(self, path, filters):
        for offset in range(0, 5000, 50):
            page = self.request("GET", path, params={**filters, "limit": 50, "offset": offset})
            rows = page["results"]
            if not isinstance(rows, list):
                raise ApiFailure("invalid pagination")
            yield from rows
            if not page.get("next") and len(rows) < 50:
                return
        raise ApiFailure("pagination limit")

    def discover(self, tournament):
        for post in self.pages("/posts/", {"tournaments": tournament, "statuses": "open",
                                           "forecast_type": list(FIELDS), "order_by": "scheduled_close_time", "with_cp": "false"}):
            yield post["id"]

    def missed_recently(self, tournament):
        cutoff = (now() - timedelta(days=1)).isoformat()
        # Server-side own-account filter, with no inspection of closed question text.
        return {p["id"] for p in self.pages("/posts/", {"tournaments": tournament,
                "statuses": ["closed", "resolved"], "forecast_type": list(FIELDS), "open_time__gt": cutoff, "with_cp": "false",
                "not_forecaster_id": self.author, "include_descriptions": "false"})}

    def detail(self, pid):
        return self.request("GET", f"/posts/{int(pid)}/", params={"with_cp": "false", "include_descriptions": "true"})

    def check_switch(self):
        if self.mode != "production":
            return
        url = ("https://api.github.com/repos/" + os.environ["GITHUB_REPOSITORY"]
               + "/actions/workflows/run_bot_on_tournament.yaml")
        response = requests.get(url, headers={"Authorization": "Bearer " + os.environ["GH_STATE_TOKEN"]},
                                timeout=8, allow_redirects=False)
        if response.status_code != 200 or response.json().get("state") != "active":
            raise ApiFailure("production publishing paused")

    def forecast(self, qid, payload):
        self.request("POST", "/questions/forecast/", json=[{"question": qid, "source": "api", **payload}])

    def comment(self, pid, text):
        self.request("POST", "/comments/create/", json={"on_post": pid, "is_private": True,
                     "included_forecast": True, "text": text, "parent": None})

    def has_comment(self, pid, marker):
        return any(marker in row["text"] for row in self.pages("/comments/", {
            "post": pid, "author": self.author, "is_private": "true"}))


def prior(post):
    personal = post["question"]["my_forecasts"]
    if not isinstance(personal["history"], list):
        raise ApiFailure("invalid account history")
    return bool(personal["history"]) or personal["latest"] is not None


def finish(api, ledger, qid, job):
    stage = job["stage"]
    pid = job["post"]
    if stage == "done":
        return "already_complete"
    if stage == "retryable":
        detail = api.detail(pid)
        q = Question.from_post(detail, respect_spot=job["mode"] == "production")
        if q.id != qid or q.post_id != pid or fingerprint(q) != job["fingerprint"]:
            return "question_changed"
        if prior(detail):
            return "previous_forecast"
        if detail["question"]["status"] != "open" or (q.close_time - now()).total_seconds() < 30:
            ledger.record(qid, stage, stage="expired")
            return "deadline_missed"
        validate_payload(q, job["payload"])
        api.check_switch()
        ledger.record(qid, stage, stage="sending")
        return deliver(api, ledger, qid, job["payload"])
    if stage == "expired":
        return "deadline_missed"
    if stage == "sending":
        detail = api.detail(pid)
        if detail["question"]["id"] != qid:
            raise StateError("recovery identity mismatch")
        if not prior(detail):
            if detail["question"]["status"] in {"closed", "resolved"}:
                ledger.record(qid, stage, stage="expired")
            return "forecast_delivery_unknown"
        ledger.record(qid, stage, stage="accepted")
        stage = "accepted"
    # The private text carries the immutable marker across process crashes.
    marker = f"[futureeval-{qid}]"
    if stage == "commenting" and api.has_comment(pid, marker):
        ledger.record(qid, stage, stage="done")
        return "completed"
    if stage not in {"accepted", "commenting"}:
        raise StateError("unexpected stage")
    ledger.record(qid, stage, stage="commenting")
    api.check_switch()
    api.comment(pid, job["comment"] + "\n\n" + marker)
    ledger.record(qid, "commenting", stage="done")
    return "completed"


def deliver(api, ledger, qid, payload):
    try:
        api.forecast(qid, payload)
    except RejectedForRateLimit:
        ledger.record(qid, "sending", stage="retryable")
        return "forecast_rate_limited"
    ledger.record(qid, "sending", stage="accepted")
    return finish(api, ledger, qid, ledger.jobs()[str(qid)])


def fingerprint(q):
    return hashlib.sha256(json.dumps([q.context(), q.grid], sort_keys=True).encode()).hexdigest()


def process(api, ledger, mode, pid, profile, generate, recovery=None):
    if recovery is not None:
        qid, job = recovery
        return finish(api, ledger, qid, job)
    post = api.detail(pid)
    if post["id"] != pid:
        raise ApiFailure("post mismatch")
    q = Question.from_post(post, respect_spot=mode == "production")
    if post["question"]["status"] != "open":
        return "closed"
    job = ledger.jobs().get(str(q.id))
    if job:
        if job["post"] != pid or job["mode"] != mode:
            raise StateError("job identity mismatch")
        return finish(api, ledger, q.id, job)
    if prior(post):
        return "previous_forecast"
    seconds = (q.close_time - now()).total_seconds()
    # Allocate at most eight minutes to research + inference. Keep a final
    # minute for re-reading the question, persisting intent, and posting.
    if seconds <= 70:
        return "deadline_missed"
    deadline = time.monotonic() + min(480, seconds - 60)
    payload, comment = generate(q, deadline, ledger, profile)
    validate_payload(q, payload)
    if not isinstance(comment, str) or not comment.strip():
        raise ValueError("missing forecast rationale")
    fresh = api.detail(pid)
    check = Question.from_post(fresh, respect_spot=mode == "production")
    if check.id != q.id or check.post_id != pid or fingerprint(check) != fingerprint(q):
        return "question_changed"
    if prior(fresh):
        return "previous_forecast"
    if fresh["question"]["status"] != "open" or (check.close_time - now()).total_seconds() < 30:
        return "deadline_missed"
    api.check_switch()
    ledger.record(q.id, None, stage="sending", post=pid, comment=comment, mode=mode,
                  payload=payload, fingerprint=fingerprint(q))
    if now() >= check.close_time:
        ledger.record(q.id, "sending", stage="expired")
        return "deadline_missed"
    return deliver(api, ledger, q.id, payload)


def run(api, ledger, mode, tournaments, profile, generate):
    counts = Counter()
    pending = {job["post"]: (int(qid), job) for qid, job in ledger.jobs().items()
               if job["stage"] not in {"done", "expired"} and job["mode"] == mode}
    posts = dict.fromkeys(pending)
    recent_misses = set()
    for tournament in tournaments:
        try:
            for pid in api.discover(tournament):
                posts.setdefault(pid)
        except Exception as exc:
            counts["discovery_failure_" + type(exc).__name__] += 1
        if mode == "production":
            try:
                recent_misses.update(api.missed_recently(tournament))
            except Exception as exc:
                counts["audit_failure_" + type(exc).__name__] += 1
    with ThreadPoolExecutor(max_workers=3) as workers:
        tasks = [workers.submit(process, api, ledger, mode, pid, profile, generate, pending.get(pid))
                 for pid in posts]
        for task in as_completed(tasks):
            try:
                counts[task.result()] += 1
            except Exception as exc:
                # Exception messages from model and API clients can contain private text.
                counts["failure_" + type(exc).__name__] += 1
    successes = {"completed", "already_complete", "previous_forecast", "closed"}
    return {"counts": dict(counts), "posts_considered": len(posts),
            "closed_without_forecast_opened_last_24h": len(recent_misses),
            "missing_or_unconfirmed": sum(n for k, n in counts.items() if k not in successes),
            "costs": ledger.costs()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dry-run", "test-area", "production", "init-state"], default="dry-run")
    args = parser.parse_args()
    if args.mode == "dry-run":
        from smoke import run as smoke_run
        print(json.dumps(smoke_run()))
        return 0
    if args.mode == "production" and os.getenv("FUTUREEVAL_ENABLED") != "true":
        print("Production is paused.")
        return 0
    if args.mode == "init-state":
        # Create-only PUT (no SHA). GitHub rejects replacement of existing content.
        GitHubFile().write(initial())
        return 0
    from pipeline import forecast
    started = time.monotonic()
    try:
        api = Metaculus(args.mode)
        api.verify_account()
        ledger = Ledger(GitHubFile(), os.getenv("PER_QUESTION_USD") or "0", os.getenv("MONTHLY_USD") or "0")
        tournaments = ["bot-testing-area"] if args.mode == "test-area" else [
            os.getenv("SEASON_ID") or "33121", os.getenv("MINIBENCH_ID") or "minibench"]
        report = run(api, ledger, args.mode, tournaments, os.getenv("FORECAST_PROFILE") or "baseline", forecast)
    except Exception as exc:
        report = {"startup_failure": type(exc).__name__, "missing_or_unconfirmed": 1}
    report["runtime_seconds"] = round(time.monotonic() - started, 2)
    report["revision"] = os.getenv("GITHUB_SHA")
    Path("health.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return int(bool(report["missing_or_unconfirmed"]))


if __name__ == "__main__":
    raise SystemExit(main())
