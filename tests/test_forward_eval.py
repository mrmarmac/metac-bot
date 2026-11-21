import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import forward_eval as fe
from smoke import MemoryStore
from state import Ledger


class ForwardEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)
        self.now = datetime(2026, 11, 21, 18, tzinfo=timezone.utc)
        self.ledger = Ledger(MemoryStore(), 1, 1)

    def capture(self):
        def response(member, messages, qid, ledger, key, deadline):
            return {"prediction": .65 if member == fe.BASELINE else .75,
                    "rationale": "Using the supplied week counts.", "citations": []}
        with patch("forward_eval.utcnow", return_value=self.now), patch("forward_eval.complete", response):
            with patch("forward_eval.sample_catalog", return_value=("https://example.org", [])):
                return fe.capture(self.path, self.ledger, "fixture")

    def test_independent_window_and_post_resolution_replay(self):
        result = self.capture()
        self.assertEqual(result["trial"], "2026-11-23")
        file = next(self.path.glob("*.capture.json"))
        with patch("forward_eval.utcnow", return_value=self.now):
            with self.assertRaises(ValueError):
                fe.resolve(file)
        snapshot = datetime(2026, 12, 1, 12, tzinfo=timezone.utc)
        with patch("forward_eval.utcnow", return_value=snapshot):
            with patch("forward_eval.sample_catalog", return_value=("https://example.org", [("a", 1)])) as read:
                row = fe.resolve(file)
                self.assertAlmostEqual(row["scores"]["experimental"]["binary_brier"], .7 ** 2)
                self.assertAlmostEqual(row["scores"]["baseline"]["binary_brier"], .65 ** 2)
                fe.resolve(file)
                self.assertEqual(read.call_count, 1)
        self.assertEqual(fe.report(self.path)["paired"], 1)
        file.write_text(file.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            fe.resolve(file)

    def test_redirection_to_new_folder_cannot_repeat_trial(self):
        self.capture()
        self.path = self.path / "second"
        with self.assertRaisesRegex(ValueError, "already claimed"):
            self.capture()

    def test_next_monday_strictly_future(self):
        monday = fe.next_window(self.now)
        self.assertEqual(fe.next_window(monday), monday + timedelta(days=7))

    def test_incomplete_catalog_fails(self):
        with patch("forward_eval.requests.get") as get:
            res = get.return_value
            res.status_code = 200
            res.content = b"{}"
            res.json.return_value = {"type": "FeatureCollection", "metadata": {"count": 1}, "features": []}
            with self.assertRaisesRegex(ValueError, "incomplete"):
                fe.sample_catalog(self.now - timedelta(days=1), self.now)
