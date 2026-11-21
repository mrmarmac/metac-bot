import json
import os
import time
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pipeline
import smoke
from state import BudgetExceeded, Ledger, StateError


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.q = smoke.sample_question("binary")
        self.now = datetime.now(timezone.utc)
        self.deadline = time.monotonic() + 60
        self.ledger = Ledger(smoke.MemoryStore(), ".10", "1")
        self.good = {"prediction": .6, "rationale": "Stable recent measurements.", "citations": []}
        self.member = pipeline.BASELINE
        self.messages = [{"role": "system", "content": "test"}, {"role": "user", "content": "é"}]

    def complete(self):
        return pipeline.complete(self.member, self.messages, 12, self.ledger, "fixture", self.deadline)

    def replay(self, fn, evidence=None):
        return pipeline.forecast_with_inputs(self.q, self.deadline, "experimental", self.now,
                                             evidence or [], "test evidence", 1, fn)

    def test_all_question_types_end_to_end(self):
        self.assertEqual(smoke.run()["model_requests"], 8)

    def test_disabled_funding_blocks_all_calls(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "fixture"}, clear=True):
            with patch("pipeline.request_json") as request, self.assertRaises(pipeline.ProviderError):
                pipeline.forecast(self.q, self.deadline, self.ledger, "baseline")
            request.assert_not_called()

    def test_no_fallback_to_personal_key(self):
        with patch.dict(os.environ, {"FORECAST_FUNDING": "competition", "OPENROUTER_API_KEY": "fixture"}, clear=True):
            with self.assertRaises(pipeline.ProviderError):
                pipeline.configured_key()

    def test_hard_budget_precedes_transport(self):
        self.ledger = Ledger(smoke.MemoryStore(), "0", "1")
        with patch("pipeline.request_json") as request, self.assertRaises(BudgetExceeded):
            self.complete()
        request.assert_not_called()

    def test_timeout_and_missing_usage_retain_maximum_reservation(self):
        with patch("pipeline.request_json", side_effect=pipeline.ProviderError("timeout")):
            with self.assertRaises(pipeline.ProviderError):
                self.complete()
        response = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(self.good)}}]}
        with patch("pipeline.request_json", return_value=response):
            self.assertEqual(self.complete(), self.good)
        self.assertEqual(len(self.ledger.data["calls"]), 2)
        self.assertTrue(all(c["amount"] > 1900 for c in self.ledger.data["calls"].values()))

    def test_rate_limit_retries_then_accounts_for_reported_usage(self):
        response = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(self.good)}}],
                    "usage": {"cost": .0003, "prompt_tokens": 103, "completion_tokens": 20}}
        with patch("pipeline.request_json", side_effect=[pipeline.HTTPFailure(429, "2"), response]) as call:
            with patch("pipeline.time.sleep") as sleep:
                self.assertEqual(self.complete(), self.good)
        sleep.assert_called_once_with(2)
        self.assertEqual(call.call_count, 2)
        amounts = [c["amount"] for c in self.ledger.data["calls"].values()]
        self.assertEqual(sorted(amounts), [0, 300])
        body = call.call_args.args[3]
        self.assertEqual(body["provider"]["only"], ["OpenAI"])
        self.assertEqual(body["provider"]["max_price"]["request"], 0)
        self.assertNotIn("tools", body)

    def test_rate_limit_past_deadline(self):
        with patch("pipeline.request_json", side_effect=pipeline.HTTPFailure(429, "300")) as call:
            with self.assertRaises(pipeline.HTTPFailure):
                self.complete()
        self.assertEqual(call.call_count, 1)

    def test_select_dated_noncommunity_sources(self):
        base = {"pub_date": "2020-01-01T00:00:00Z", "article_url": "https://example.com/news",
                "summary": "A dated report.", "eng_title": "Observation"}
        rows = [base, dict(base), dict(base, article_url="https://www.metaculus.com/guess"),
                dict(base, pub_date="2020-01-01"), dict(base, pub_date="2099-01-01T00:00:00Z"),
                dict(base, article_url="https://example.gov/report")]
        records = pipeline.select_evidence(rows, self.now)
        self.assertEqual([r.url for r in records], ["https://example.gov/report", "https://example.com/news"])

    def test_no_live_retrieval_during_historical_replay(self):
        item = pipeline.Evidence(1, "2099-01-01T00:00:00+00:00", "https://example.org", "Future", "x")
        with patch("pipeline.request_json") as request, self.assertRaises(ValueError):
            self.replay(Mock(), [item])
        request.assert_not_called()

    def test_unmatched_citation_or_model_url_rejected(self):
        for invalid in [dict(self.good, citations=[{"id": 2, "quote": "hello"}]),
                        dict(self.good, rationale="See https://invented.example.")]:
            with self.subTest(answer=invalid), self.assertRaises(pipeline.ProviderError):
                pipeline.explanation(invalid, [])

    def test_distinct_methods_and_budget_fallback(self):
        model = Mock(side_effect=[self.good, BudgetExceeded("limit")])
        result, comment = self.replay(model)
        self.assertAlmostEqual(result["probability_yes"], .6)
        self.assertIn("1 of 2 estimates", comment)
        self.assertNotEqual(model.call_args_list[0].args[1][0], model.call_args_list[1].args[1][0])

    def test_journal_failure_propagates(self):
        model = Mock(side_effect=[self.good, StateError("cannot persist")])
        with self.assertRaises(StateError):
            self.replay(model)

    def test_unknown_predictions_do_not_silently_create_fallback(self):
        model = Mock(side_effect=[dict(self.good, prediction="Yes"), pipeline.ProviderError("timeout")])
        with self.assertRaises(pipeline.ProviderError):
            self.replay(model)

    def test_tavily_forecast_cites_only_eligible_dated_snippets(self):
        result = {"url": "https://example.gov/release", "title": "Sensor release",
                  "content": "Stable recent measurements.", "published_date": "Tue, 11 Mar 2025 17:00:00 GMT"}
        rows = [result, result, dict(result, published_date=None),
                dict(result, url="https://metaculus.com/latest"),
                dict(result, published_date="2099-01-01T00:00:00Z")]
        answer = dict(self.good, citations=[{"id": 1, "quote": result["content"]}])
        completion = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer)}}]}
        env = {"TAVILY_API_KEY": "news-key", "FORECAST_FUNDING": "paid", "OPENROUTER_API_KEY": "model-key"}
        with patch.dict(os.environ, env, clear=True), patch("pipeline.request_json",
                side_effect=[{"results": rows}, completion]) as request:
            payload, comment = pipeline.forecast(self.q, self.deadline, self.ledger, "baseline")
        search, model = request.call_args_list
        self.assertEqual(search.args[:3], ("api.tavily.com", "/search", "news-key"))
        self.assertEqual(search.args[3]["search_depth"], "basic")
        self.assertFalse(search.args[3]["auto_parameters"])
        sources = json.loads(model.args[3]["messages"][1]["content"])["article_summaries"]
        self.assertEqual([s["url"] for s in sources], [result["url"]])
        self.assertEqual(payload["probability_yes"], .6)
        self.assertIn("Tavily", comment)
        self.assertIn(result["url"], comment)

    def test_tavily_failure_deadline_and_asknews_precedence(self):
        env = {"TAVILY_API_KEY": "fixture"}
        with patch.dict(os.environ, env, clear=True), patch("pipeline.request_json",
                  side_effect=pipeline.ProviderError("unavailable")) as request:
            evidence, note = pipeline.research(self.q, self.deadline, self.now)
            self.assertEqual(evidence, [])
            self.assertIn("Tavily", note)
            pipeline.research(self.q, time.monotonic() + 3, self.now)
            request.assert_called_once()
            with patch.dict(os.environ, {"ASKNEWS_COMPETITION_ACCESS": "true", "ASKNEWS_API_KEY": "fixture"}):
                pipeline.research(self.q, self.deadline, self.now)
            self.assertEqual(request.call_args.args[0], "api.asknews.app")
