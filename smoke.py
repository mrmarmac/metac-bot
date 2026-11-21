"""Credential-free rehearsal of the research, inference, and forecast formatting paths."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pipeline
from formats import Question, validate_payload
from state import Ledger, initial


class MemoryStore:
    def __init__(self):
        self.data = initial()

    def read(self):
        return self.data

    def write(self, data):
        self.data = data


def sample_question(kind: str, index: int = 1) -> Question:
    """Generated practice questions with no corresponding Metaculus post."""
    now = datetime.now(timezone.utc)
    start, close, resolve = (now + delta for delta in
                             [timedelta(hours=-1), timedelta(hours=2), timedelta(days=7)])
    question = {"id": 1000 + index, "type": kind, "title": f"Synthetic sensor exercise ({kind})",
                "description": "Practice sensor A reports a value between 0 and 100.",
                "resolution_criteria": "Read the synthetic sensor at the scheduled resolution time.",
                "fine_print": "The practice sensor reports in units.", "unit": "units",
                "open_time": start.isoformat(), "scheduled_close_time": close.isoformat(),
                "actual_close_time": None, "scheduled_resolve_time": resolve.isoformat(),
                "status": "open", "options": ["low", "high"] if kind == "multiple_choice" else None}
    if kind in ("numeric", "discrete"):
        question.update(scaling={"range_min": 0, "range_max": 100, "zero_point": None,
                                 "inbound_outcome_count": 101 if kind == "discrete" else 200},
                        open_lower_bound=False, open_upper_bound=False)
    return Question.from_post({"id": index, "question": question, "title": question["title"],
                               "open_time": start.isoformat(), "scheduled_close_time": close.isoformat(),
                               "status": "open", "scheduled_resolve_time": resolve.isoformat()})


def run() -> dict:
    store = MemoryStore()
    ledger = Ledger(store, per_question="0.10", monthly="1")
    samples = {"binary": 0.45, "multiple_choice": {"low": 0.55, "high": 0.45},
               "numeric": [15, 30, 50, 70, 85], "discrete": [15, 30, 50, 70, 85]}
    counts = {"research_requests": 0, "model_requests": 0}
    with patch.dict(os.environ, {"FORECAST_FUNDING": "competition",
                                 "METACULUS_OPENROUTER_KEY": "fixture",
                                 "ASKNEWS_COMPETITION_ACCESS": "true",
                                 "ASKNEWS_API_KEY": "fixture", "FORECAST_CALIBRATION": "1",
                                 "LITELLM_LOCAL_MODEL_COST_MAP": "True"}, clear=True):
        for index, (kind, prediction) in enumerate(samples.items(), 1):
            q = sample_question(kind, index)

            def transport(host, path, key, body, deadline):
                if host == "api.asknews.app":
                    counts["research_requests"] += 1
                    return {"as_dicts": [{"article_url": "https://example.org/sensor",
                                          "eng_title": "Practice observations",
                                          "summary": "Synthetic observations center on 50 units.",
                                          "pub_date": "2020-01-01T00:00:00Z"}]}
                assert host == "openrouter.ai"
                counts["model_requests"] += 1
                answer = {"prediction": prediction, "rationale": "The synthetic series centers on 50.",
                          "citations": [{"id": 1, "quote": "center on 50 units"}]}
                return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer)}}],
                        "usage": {"cost": 0.001, "prompt_tokens": 100, "completion_tokens": 20}}

            with patch.object(pipeline, "request_json", transport):
                payload, comment = pipeline.forecast(q, time.monotonic() + 30, ledger, "experimental")
            validate_payload(q, payload)
            if "2 of 2 estimates" not in comment or "example.org/sensor" not in comment:
                raise RuntimeError("incomplete smoke forecast")
    return {"mode": "synthetic", "formats": list(samples), "cash_usd": 0, **counts}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
