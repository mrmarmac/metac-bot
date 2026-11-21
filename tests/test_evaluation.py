import math
import unittest
from datetime import date, timedelta
from types import SimpleNamespace

from evaluation import Decision, calibration_bins, evaluate, history_at, load_decisions, make_question, predict, score


class EvaluationTests(unittest.TestCase):
    def test_forecast_invariant_to_future_releases(self):
        rows = load_decisions()
        meeting = date(2024, 9, 18)
        cutoff = meeting - timedelta(days=1)
        poisoned = [Decision(row.published, 900) if row.published >= cutoff else row for row in rows]
        question = make_question("binary", cutoff, meeting)
        last = max(row.published for row in rows if row.published < cutoff)
        for profile in ("baseline", "experimental"):
            self.assertEqual(
                predict(question, history_at(rows, cutoff), profile, cutoff, last),
                predict(question, history_at(poisoned, cutoff), profile, cutoff, last),
            )
        self.assertEqual(history_at([Decision(cutoff, 99)], cutoff), ())

    def test_binary_and_category_losses(self):
        q = SimpleNamespace(kind="binary")
        result = score(q, {"probability_yes": 0.8}, 0)
        self.assertAlmostEqual(result["binary_brier"], 0.04)
        self.assertAlmostEqual(result["binary_log_loss"], -math.log(.8))
        q = SimpleNamespace(kind="multiple_choice")
        result = score(q, {"probability_yes_per_category": {"Raise": .6, "Hold": .3, "Cut": .1}}, -25)
        self.assertAlmostEqual(result["multiple_choice_log_loss"], -math.log(.1))

    def test_integral_uniform_cdf_and_discrete_rps(self):
        q = SimpleNamespace(kind="numeric", grid=(-1., 0., 1.))
        self.assertAlmostEqual(score(q, {"continuous_cdf": [0, .5, 1]}, 0)["numeric_crps_bp"], 1 / 6)
        q = SimpleNamespace(kind="discrete", open_lower_bound=False, open_upper_bound=False, grid=(-37.5, -12.5, 12.5, 37.5))
        self.assertEqual(score(q, {"continuous_cdf": [0, .25, .75, 1]}, 0)["discrete_normalized_rps"], .0625)

    def test_reliability_bins(self):
        bins = calibration_bins([(0.2, 1), (0.3, 0), (1, 1)])
        self.assertEqual([b["n"] for b in bins], [2, 1])
        self.assertEqual(bins[0]["event_frequency"], .5)
        self.assertEqual(bins[0]["mean_prediction"], .25)

    def test_meeting_coverage_and_expected_scores(self):
        for profile in ("baseline", "experimental"):
            result = evaluate(load_decisions(), profile, 1.0)
            self.assertEqual(result["failures"], [])
            self.assertEqual(result["scored"], 64)
            self.assertEqual(result["api_cost_usd"], 0)
            self.assertEqual(len(result["losses_lower_is_better"]), 5)
