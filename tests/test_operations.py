import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import os
import threading
import unittest
from unittest.mock import Mock, patch

from cryptography.fernet import Fernet

from runner import ApiFailure, Metaculus, RejectedForRateLimit, prior, run
from state import BudgetExceeded, GitHubFile, Ledger, StateError, initial


class Store:
    def __init__(self):
        self.data = initial()

    def read(self):
        return copy.deepcopy(self.data)

    def write(self, data):
        self.data = copy.deepcopy(data)


class Platform:
    def __init__(self):
        self.personal = {"history": [], "latest": None}
        self.ending = datetime.now(timezone.utc) + timedelta(hours=1)
        self.criteria = "Sensor reads above zero at noon."
        self.writes = 0
        self.comments = []
        self.ids = [1]
        self.drop_forecast = self.drop_comment = self.reject = False

    def check_switch(self):
        pass

    def discover(self, tournament):
        return self.ids

    def missed_recently(self, tournament):
        return {66}

    def detail(self, pid):
        return {"id": pid, "question": {"id": 100, "status": "open", "type": "binary",
                "title": "Sensor exercise", "resolution_criteria": self.criteria,
                "scheduled_close_time": self.ending.isoformat(), "my_forecasts": copy.deepcopy(self.personal)}}

    def forecast(self, qid, payload):
        if self.reject:
            raise RejectedForRateLimit()
        self.writes += 1
        self.personal["latest"] = {"forecast_values": [0.6, 0.4]}
        if self.drop_forecast:
            raise ApiFailure()

    def comment(self, pid, text):
        self.comments.append(text)
        if self.drop_comment:
            raise ApiFailure()

    def has_comment(self, pid, marker):
        return any(marker in c for c in self.comments)


class Operations(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.api = Platform()
        self.generate = Mock(return_value=({"probability_yes": 0.4}, "Synthetic rationale."))

    def cycle(self):
        # Fresh ledger object models a new ephemeral Actions runner.
        return run(self.api, Ledger(self.store, ".1", "1"), "production",
                   ["33121", "minibench"], "baseline", self.generate)

    def test_discovery_overlap_and_repeated_run(self):
        report = self.cycle()
        self.assertEqual(report["counts"], {"completed": 1})
        self.assertEqual(report["closed_without_forecast_opened_last_24h"], 1)
        self.assertEqual(self.cycle()["counts"], {"already_complete": 1})
        self.assertEqual(self.api.writes, 1)
        self.assertEqual(len(self.api.comments), 1)

    def test_remote_state_even_with_no_local_job(self):
        self.api.personal["history"] = [{"end_time": "2024-01-01T00:00:00Z"}]
        self.assertEqual(self.cycle()["counts"], {"previous_forecast": 1})
        self.generate.assert_not_called()

    def test_unavailable_remote_state_fails_closed(self):
        del self.api.personal["history"]
        self.assertEqual(self.cycle()["missing_or_unconfirmed"], 1)
        self.generate.assert_not_called()

    def test_lost_forecast_receipt_then_closed_recovers_comment(self):
        self.api.drop_forecast = True
        self.assertEqual(self.cycle()["missing_or_unconfirmed"], 1)
        self.api.ids = []
        self.assertEqual(self.cycle()["counts"], {"completed": 1})
        self.assertEqual(self.api.writes, 1)
        self.generate.assert_called_once()

    def test_ambiguous_forecast_never_resent(self):
        self.api.forecast = Mock(side_effect=ApiFailure())
        self.cycle()
        self.assertEqual(self.cycle()["counts"], {"forecast_delivery_unknown": 1})
        self.api.forecast.assert_called_once()
        self.assertFalse(self.api.comments)

    def test_known_rejection_can_retry_without_inference(self):
        self.api.reject = True
        self.assertEqual(self.cycle()["counts"], {"forecast_rate_limited": 1})
        self.api.reject = False
        self.assertEqual(self.cycle()["counts"], {"completed": 1})
        self.generate.assert_called_once()

    def test_lost_comment_receipt_uses_private_marker(self):
        self.api.drop_comment = True
        self.cycle()
        self.assertEqual(self.cycle()["counts"], {"completed": 1})
        self.assertEqual(len(self.api.comments), 1)
        self.assertEqual(self.api.writes, 1)

    def test_changed_or_expired_never_posts(self):
        def mutate(*args):
            self.api.criteria += "Different threshold."
            return {"probability_yes": 0.4}, "fixture"
        self.generate.side_effect = mutate
        self.assertEqual(self.cycle()["counts"], {"question_changed": 1})
        self.api.ending = datetime.now(timezone.utc) + timedelta(seconds=5)
        self.assertEqual(self.cycle()["counts"], {"deadline_missed": 1})
        self.assertEqual(self.api.writes, 0)

    def test_private_comment_api_shape(self):
        with patch.dict(os.environ, {"METACULUS_TOKEN": "fixture", "METACULUS_BOT_ID": "77"}):
            client = Metaculus()
        client.request = Mock()
        client.comment(123, "test")
        data = client.request.call_args.kwargs["json"]
        self.assertTrue(data["is_private"] and data["included_forecast"])
        self.assertEqual(data["on_post"], 123)


class Budget(unittest.TestCase):
    def test_limits_and_unknown_usage_survive_restart(self):
        store = Store()
        ledger = Ledger(store, ".01", ".02")
        key = ledger.reserve(1, ".008")
        with self.assertRaises(BudgetExceeded):
            ledger.reserve(1, ".003")
        ledger.settle(key, ".003", 12, 9)
        ledger.reserve(1, ".007")
        ledger.reserve(2, ".01")
        with self.assertRaises(BudgetExceeded):
            Ledger(store, ".01", ".02").reserve(3, ".000001")
        self.assertEqual(ledger.costs()["outstanding_reserved_usd"], .017)

    def test_concurrent_budget_reservations(self):
        ledger = Ledger(Store(), "1", ".01")
        def attempt(i):
            try:
                ledger.reserve(i, ".006")
                return 1
            except BudgetExceeded:
                return 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(sum(pool.map(attempt, range(8))), 1)

    def test_state_failure_freezes_journal(self):
        store = Store()
        ledger = Ledger(store, "1", "1")
        store.write = Mock(side_effect=OSError())
        with self.assertRaises(OSError):
            ledger.reserve(1, ".001")
        with self.assertRaises(StateError):
            ledger.reserve(1, ".001")
        self.assertEqual(store.read(), initial())

    def test_encrypted_state_uses_conditional_sha(self):
        env = {"STATE_KEY": Fernet.generate_key().decode(), "GH_STATE_TOKEN": "fixture",
               "GITHUB_REPOSITORY": "example/test"}
        with patch.dict(os.environ, env), patch("state.requests.put") as put, patch("state.requests.get") as get:
            file = GitHubFile()
            put.return_value.status_code = 201
            put.return_value.json.return_value = {"content": {"sha": "first"}}
            file.write({"secret": "synthetic private text"})
            body = put.call_args.kwargs["json"]
            self.assertNotIn("sha", body)
            self.assertNotIn("synthetic", body["content"])
            get.return_value.status_code = 200
            get.return_value.json.return_value = {"sha": "first", "content": body["content"]}
            self.assertEqual(file.read()["secret"], "synthetic private text")
            put.return_value.status_code = 409
            with self.assertRaises(StateError):
                file.write(initial())
            self.assertEqual(put.call_args.kwargs["json"]["sha"], "first")

    def test_zero_default_and_nonfinite_values(self):
        ledger = Ledger(Store(), 0, 0)
        with self.assertRaises(BudgetExceeded):
            ledger.reserve(1, ".0000001")
        for x in ["Infinity", "NaN", -1]:
            with self.assertRaises(ValueError):
                ledger.reserve(1, x)
