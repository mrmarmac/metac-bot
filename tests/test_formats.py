import asyncio
import contextlib
import io
import math
import random
import socket
import unittest
from unittest.mock import AsyncMock, patch

from formats import Question, aggregate, display_forecast, prediction_payload, validate_payload


def post(kind, **overrides):
    question = {
        "id": 19, "title": "Independent synthetic observation", "type": kind,
        "description": "A simulated counter.", "resolution_criteria": "Use sensor A at midnight.",
        "fine_print": "Ignore sensor B.", "unit": "items",
        "scheduled_close_time": "2020-01-02T03:00:00+03:00",
        "scheduled_resolve_time": "2020-01-04T00:00:00Z",
        "options": ["Low", "High"] if kind == "multiple_choice" else [],
        "scaling": {"range_min": 0, "range_max": 10, "zero_point": None,
                    "inbound_outcome_count": 10 if kind == "discrete" else 200},
        "open_lower_bound": False, "open_upper_bound": False,
        **overrides,
    }
    return {"id": 15, "question": question}


def question(kind, **overrides):
    return Question.from_post(post(kind, **overrides))


class FormatTests(unittest.TestCase):
    def test_target_and_context(self):
        q = question("numeric", aggregations={"unweighted": 0.23}, my_forecasts={"latest": "hidden"},
                     actual_close_time="2020-01-01T21:00:00Z")
        self.assertEqual((q.id, q.post_id), (19, 15))
        self.assertEqual(q.close_time.isoformat(), "2020-01-01T21:00:00+00:00")
        context = q.context()
        self.assertEqual(context["unit"], "items")
        self.assertEqual(context["resolution_criteria"], "Use sensor A at midnight.")
        self.assertEqual(context["scheduled_resolve_time"], "2020-01-04T00:00:00Z")
        self.assertNotIn("aggregations", context)
        self.assertNotIn("my_forecasts", context)
        for changes in ({"type": "conditional"}, {"scheduled_close_time": "2020-01-01"},
                        {"post_id": 77}, {"resolution_criteria": None}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                question("binary", **changes)

    def test_scoring_cutoff_and_mc_bounds_match_server(self):
        q = question("multiple_choice", spot_scoring_time="2020-01-01T23:15:00Z")
        self.assertEqual(q.close_time.isoformat(), "2020-01-01T23:15:00+00:00")
        with self.assertRaises(ValueError):
            validate_payload(q, {"probability_yes_per_category": {"Low": 0.0009999999999, "High": 0.9990000000001}})

    def test_binary_and_calibration(self):
        q = question("binary")
        a, b = [prediction_payload(q, p) for p in (0.8, 0.6)]
        result = aggregate(q, [a, b])
        self.assertAlmostEqual(result["probability_yes"], 0.7)
        self.assertLess(aggregate(q, [a, b], 0.5)["probability_yes"], 0.7)
        self.assertEqual(prediction_payload(q, 0), {"probability_yes": 0.001})
        self.assertEqual(prediction_payload(q, 1), {"probability_yes": 0.999})
        for p in (-0.01, 1.1, math.inf, math.nan, True, "0.3"):
            with self.subTest(p=p), self.assertRaises(ValueError):
                prediction_payload(q, p)
        for p in ({"probability_yes": 0}, {"probability_yes": 1}, {"probability_yes": 0.5, "question": 20}):
            with self.assertRaises(ValueError):
                validate_payload(q, p)
        for forecasts, factor in (([], 1), ([a], 0), ([a], math.nan)):
            with self.assertRaises(ValueError):
                aggregate(q, forecasts, factor)

    def test_mc_matches_labels_and_sum(self):
        q = question("multiple_choice")
        x = prediction_payload(q, {"High": 1, "Low": 0})
        y = prediction_payload(q, {"Low": 0.2, "High": 0.8})
        self.assertEqual(x["probability_yes_per_category"], {"Low": 0.001, "High": 0.999})
        self.assertAlmostEqual(aggregate(q, [x, y])["probability_yes_per_category"]["Low"], 0.1005)
        for raw in ({"High": 1}, {"High": .5, "low": .5}, {"High": .2, "Low": .2},
                    {"Low": -1, "High": 2}, {"High": .5, "Low": math.nan}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                prediction_payload(q, raw)
        with self.assertRaises(ValueError):
            validate_payload(q, {"probability_yes_per_category": {"Low": 0, "High": 1}})
        with self.assertRaises(ValueError):
            question("multiple_choice", options=["a", "a"])

    def test_linear_log_and_explicit_grids(self):
        q = question("numeric", scaling={"range_min": 1, "range_max": 100, "zero_point": 0})
        self.assertEqual(len(q.grid), 201)
        self.assertAlmostEqual(q.grid[100], 10.0)
        cdf = prediction_payload(q, [3, 5, 10, 20, 30])["continuous_cdf"]
        self.assertAlmostEqual(cdf[100], 0.5, places=5)
        q = question("discrete", scaling={"range_min": -0.5, "range_max": 5.5,
                                          "continuous_range": [-0.5, 0.5, 2.5, 5.5]})
        self.assertEqual(q.grid, (-0.5, 0.5, 2.5, 5.5))
        payload = prediction_payload(q, [1.5] * 5)
        self.assertIn("P50 = 1.5", display_forecast(q, payload))
        self.assertEqual(len(payload["continuous_cdf"]), len(q.grid))
        for scaling in ({"range_min": 0, "range_max": 10},
                        {"range_min": 0, "range_max": 10, "inbound_outcome_count": 3, "continuous_range": [0, 4, 4, 10]},
                        {"range_min": 1, "range_max": 10, "inbound_outcome_count": 4, "zero_point": 3}):
            with self.assertRaises(ValueError):
                question("discrete", scaling=scaling)

    def test_numeric_and_discrete_tails_mass_and_pool(self):
        for kind in ("numeric", "discrete"):
            for lo, hi in ((False, False), (True, True), (True, False), (False, True)):
                q = question(kind, open_lower_bound=lo, open_upper_bound=hi)
                for nominal in ([0, 0, 0, 0, 0], [10] * 5,
                                [-5 if lo else 0, 2, 4, 6, 15 if hi else 10]):
                    with self.subTest(kind=kind, lo=lo, hi=hi, nominal=nominal):
                        payload = prediction_payload(q, nominal)
                        cdf = payload["continuous_cdf"]
                        n = len(q.grid) - 1
                        self.assertEqual(len(cdf), n + 1)
                        self.assertTrue(cdf[0] >= 0.001 if lo else cdf[0] == 0)
                        self.assertTrue(cdf[-1] <= 0.999 if hi else cdf[-1] == 1)
                        for a, b in zip(cdf, cdf[1:]):
                            self.assertGreaterEqual(b - a, 0.01 / n)
                            self.assertLessEqual(b - a, 40 / n)
                        other = prediction_payload(q, [1, 2, 3, 4, 5])
                        pooled = aggregate(q, [payload, other])["continuous_cdf"]
                        self.assertAlmostEqual(pooled[2], (cdf[2] + other["continuous_cdf"][2]) / 2)

    def test_server_rounded_pmf_constraints(self):
        # Metaculus/questions/serializers/common.py rounds CDF to 10 decimals
        # and its first differences to 9 before comparing to these thresholds.
        import numpy as np

        rng = random.Random(11)
        for n in (1, 2, 3, 16, 17, 71, 99, 199, 200, 201, 255, 1000):
            for lo, hi in ((True, True), (False, False), (True, False), (False, True)):
                q = question("numeric", open_lower_bound=lo, open_upper_bound=hi,
                             scaling={"range_min": -100, "range_max": 100, "inbound_outcome_count": n})
                for i in range(20):
                    choices = range(-300 if lo else -100, 301 if hi else 101)
                    xs = sorted(rng.choices(choices, k=5)) if i % 2 else [23] * 5
                    cdf = np.round(prediction_payload(q, xs)["continuous_cdf"], 10)
                    increments = np.round(np.diff(cdf), 9)
                    with self.subTest(n=n, lo=lo, hi=hi, i=i):
                        self.assertEqual(len(cdf), n + 1)
                        self.assertTrue(np.all(increments >= np.round(0.01 / n, 9)))
                        self.assertTrue(np.all(increments <= 40 / n))
                        self.assertTrue(cdf[0] >= 0.001 if lo else cdf[0] == 0)
                        self.assertTrue(cdf[-1] <= 0.999 if hi else cdf[-1] == 1)

    def test_reject_bad_numeric_payloads(self):
        q = question("numeric")
        good = prediction_payload(q, [1, 2, 3, 4, 5])["continuous_cdf"]
        for index, value in ((0, .1), (-1, .999), (100, good[99]), (120, 0.99), (87, math.nan)):
            broken = good.copy()
            broken[index] = value
            with self.assertRaises(ValueError):
                validate_payload(q, {"continuous_cdf": broken})
        for raw in ([1, 2, 3, 2, 4], [1, 2, 3, 4, 11], [1, 2, 3, 4], [1, 2, 3, 4, True]):
            with self.assertRaises(ValueError):
                prediction_payload(q, raw)
        with self.assertRaises(ValueError):
            validate_payload(q, {"continuous_cdf": good[:-1]})
        with self.assertRaises(ValueError):
            validate_payload(question("numeric", open_lower_bound=True), {"continuous_cdf": good})


class OfficialStarterTests(unittest.TestCase):
    def test_four_synthetic_reports_without_network(self):
        with patch.object(socket.socket, "connect", side_effect=AssertionError("network disabled")):
            import main_with_no_framework as starter
            replies = {
                "binary": "Probability: 65%", "multiple_choice": "Low: 60\nHigh: 40",
                "numeric": "Percentile 10: 2\nPercentile 50: 5\nPercentile 90: 8",
                "discrete": "Percentile 10: 2\nPercentile 50: 5\nPercentile 90: 8",
            }
            for kind, reply in replies.items():
                with (self.subTest(kind=kind), patch.object(starter, "get_post_details", return_value=post(kind)),
                      patch.object(starter, "run_research", return_value="Synthetic fixture only"),
                      patch.object(starter, "call_llm", new=AsyncMock(return_value=reply)),
                      contextlib.redirect_stdout(io.StringIO())):
                    report = asyncio.run(starter.forecast_individual_question(19, 15, False, 1, False))
                    self.assertIn("Forecast:", report)
                    self.assertIn("Comment:", report)
