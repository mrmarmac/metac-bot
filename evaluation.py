"""Walk-forward reference-class forecast evaluation on public FOMC decisions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time as clock_time, timedelta, timezone
from pathlib import Path
from statistics import NormalDist, fmean, pstdev

from formats import Question, validate_payload
from pipeline import BASELINE, Evidence, ProviderError, forecast_with_inputs

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "fixtures" / "fomc_decisions.csv"
HOLDOUT = date(2024, 1, 1)
KINDS = ("binary", "multiple_choice", "numeric", "discrete")
OPTIONS = ("Cut", "Hold", "Raise")
PROFILES = {"baseline": (None,), "experimental": (None, 4)}


@dataclass(frozen=True)
class Decision:
    published: date
    change_bp: int


def load_decisions(path: Path = DATA) -> list[Decision]:
    with path.open(newline="") as stream:
        rows = [
            Decision(date.fromisoformat(row["announcement_date"]), int(row["change_bp"]))
            for row in csv.DictReader(stream)
        ]
    dates = [row.published for row in rows]
    if dates != sorted(set(dates)):
        raise ValueError("Decisions must be unique and in chronological order")
    return rows


def history_at(rows: list[Decision], cutoff: date) -> tuple[int, ...]:
    # The cutoff is start-of-day UTC. Only strictly earlier release dates qualify.
    return tuple(row.change_bp for row in rows if row.published < cutoff)


def direction(change: int) -> str:
    if change == 0:
        return "Hold"
    return "Raise" if change > 0 else "Cut"


def make_question(kind: str, cutoff: date, meeting: date) -> Question:
    titles = {
        "binary": "Will the next scheduled FOMC decision keep the target range unchanged?",
        "multiple_choice": "Which direction will the next scheduled FOMC decision take?",
        "numeric": "What is the signed change in the target range upper endpoint, in basis points?",
        "discrete": "To the nearest 25 basis points, what is the signed change in the upper endpoint?",
    }
    # A local fixture shaped like the API. These IDs never enter a network call.
    identifier = meeting.toordinal() * 10 + KINDS.index(kind)
    question = {
        "id": identifier,
        "title": titles[kind],
        "type": kind,
        "description": "Scheduled US monetary policy announcements.",
        "resolution_criteria": f"Use the {meeting.isoformat()} Federal Reserve statement. One basis point is 0.01 percentage points.",
        "fine_print": "Numeric changes are clipped at ±112.5 bp. Discrete changes are rounded to the nearest 25 bp and clipped at ±100 bp.",
        "open_time": (cutoff - timedelta(days=1)).isoformat() + "T00:00:00Z",
        "scheduled_close_time": cutoff.isoformat() + "T03:00:00Z",
        "scheduled_resolve_time": meeting.isoformat() + "T23:59:59Z",
        "options": list(OPTIONS) if kind == "multiple_choice" else [],
        "scaling": {
            "range_min": -112.5, "range_max": 112.5, "zero_point": None,
            "inbound_outcome_count": 9 if kind == "discrete" else 200,
        },
        "open_lower_bound": False,
        "open_upper_bound": False,
        "unit": "basis points",
    }
    return Question.from_post({"id": identifier, "question": question})


def estimate(q: Question, history: tuple[int, ...]) -> float | dict | list:
    if not history:
        raise ValueError("A reference class needs at least one prior decision")
    if q.kind in ("binary", "multiple_choice"):
        counts = Counter(direction(change) for change in history)
        # One pseudocount per outcome, including unseen categories.
        if q.kind == "binary":
            raw = (counts["Hold"] + 1) / (len(history) + 2)
        else:
            raw = {option: (counts[option] + 1) / (len(history) + 3) for option in OPTIONS}
    else:
        normal = NormalDist(fmean(history), max(12.5, pstdev(history)))
        lo, hi = normal.cdf(q.grid[0]), normal.cdf(q.grid[-1])
        raw = [normal.inv_cdf(lo + (hi - lo) * p) for p in (.1, .25, .5, .75, .9)]
    return raw


def predict(q: Question, history: tuple[int, ...], profile: str, cutoff: date,
            last_release: date, calibration: float = 1.0) -> dict:
    change = history[-1]
    summary = f"The announced target upper endpoint change was {change} basis points."
    url = f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{last_release:%Y%m%d}a.htm"
    evidence = [Evidence(1, last_release.isoformat() + "T23:59:59+00:00", url, "FOMC statement", summary)]
    now = datetime.combine(cutoff, clock_time(), timezone.utc)

    def reference_class_member(member, messages, member_deadline):
        # A drop-in deterministic replacement for inference. Present-day model
        # weights and current web search would both contaminate a historical replay.
        window = PROFILES[profile][0 if member == BASELINE else 1]
        series = history if window is None else history[-window:]
        return {
            "prediction": estimate(q, series),
            "rationale": f"Offline reference class of {len(series)} prior scheduled decisions. "
                         f"The mean signed change was {fmean(series):.2f} basis points.",
            "citations": [{"id": 1, "quote": summary}],
        }

    payload, comment = forecast_with_inputs(
        q, time.monotonic() + 30, profile, now, evidence, "Offline FOMC replay.",
        calibration, reference_class_member,
    )
    if url not in comment:
        raise ValueError("Missing verified statement in explanation")
    validate_payload(q, payload)
    return payload


def score(q: Question, payload: dict, change: int) -> dict[str, float]:
    if q.kind == "binary":
        outcome = change == 0
    elif q.kind == "multiple_choice":
        outcome = direction(change)
    elif q.kind == "discrete":
        outcome = min(100, max(-100, 25 * round(change / 25)))
    else:
        outcome = min(q.grid[-1], max(q.grid[0], change))
    scores = score_outcome(q, payload, outcome)
    if "numeric_bounded_crps" in scores:
        return {"numeric_crps_bp": scores["numeric_bounded_crps"]}
    return scores


def score_outcome(q: Question, payload: dict, outcome: bool | str | float) -> dict[str, float]:
    """Score a validated payload. Numeric integrals are restricted to question bounds."""
    if q.kind == "binary":
        if type(outcome) is not bool:
            raise ValueError("binary resolution must be a Boolean")
        p = payload["probability_yes"]
        return {"binary_brier": (p - outcome) ** 2,
                "binary_log_loss": -math.log(p if outcome else 1 - p)}
    if q.kind == "multiple_choice":
        choices = payload["probability_yes_per_category"]
        if not isinstance(outcome, str) or outcome not in choices:
            raise ValueError("resolution must exactly match an option")
        return {"multiple_choice_log_loss": -math.log(choices[outcome])}
    if type(outcome) not in (int, float) or not math.isfinite(outcome):
        raise ValueError("distribution resolution must be a finite number")
    edges = q.grid
    cdf = payload["continuous_cdf"]
    if outcome < edges[0] and not q.open_lower_bound or outcome > edges[-1] and not q.open_upper_bound:
        raise ValueError("resolution outside a closed bound")
    if q.kind == "discrete":
        # Interior edges separate adjacent outcomes. Include the boundary cutoffs
        # when underflow or overflow is a possible outcome.
        indices = range(0 if q.open_lower_bound else 1,
                        len(edges) if q.open_upper_bound else len(edges) - 1)
        return {"discrete_normalized_rps": fmean(
            (cdf[i] - (outcome < edges[i] if i < len(edges) - 1 else outcome <= edges[i])) ** 2
            for i in indices
        )}
    # Integrate squared CDF error exactly for the piecewise-linear submitted CDF.
    # Split a bin at the observed value before integrating the step in the target.
    loss = 0.0
    for left, right, p, r in zip(edges[:-1], edges[1:], cdf[:-1], cdf[1:], strict=True):
        split = min(right, max(left, outcome))
        mid = p + (r - p) * (split - left) / (right - left)
        loss += (split - left) * (p * p + p * mid + mid * mid) / 3
        a, b = 1 - mid, 1 - r
        loss += (right - split) * (a * a + a * b + b * b) / 3
    return {"numeric_bounded_crps": loss}


def calibration_bins(observations: list[tuple[float, int]]) -> list[dict]:
    bins = defaultdict(list)
    for p, y in observations:
        bins[min(4, int(p * 5))].append((p, y))
    return [
        {"interval": [index / 5, (index + 1) / 5], "n": len(items),
         "mean_prediction": fmean(p for p, _ in items), "event_frequency": fmean(y for _, y in items)}
        for index, items in sorted(bins.items())
    ]


def evaluate(rows: list[Decision], profile: str, calibration: float) -> dict:
    started = time.perf_counter()
    losses = defaultdict(list)
    observations = []
    failures = []
    scored = 0
    holdout = [row for row in rows if row.published >= HOLDOUT]
    for row in holdout:
        cutoff = row.published - timedelta(days=1)
        history = history_at(rows, cutoff)
        last_release = max(release.published for release in rows if release.published < cutoff)
        for kind in KINDS:
            try:
                question = make_question(kind, cutoff, row.published)
                payload = predict(question, history, profile, cutoff, last_release, calibration)
                for name, value in score(question, payload, row.change_bp).items():
                    losses[name].append(value)
                if kind == "binary":
                    observations.append((payload["probability_yes"], int(row.change_bp == 0)))
                scored += 1
            except (ValueError, TypeError, KeyError, ArithmeticError, ProviderError) as error:
                failures.append({"meeting": row.published.isoformat(), "kind": kind, "error": str(error)})
    eligible = len(holdout) * len(KINDS)
    return {
        "windows_in_meetings": PROFILES[profile], "calibration": calibration, "eligible": eligible,
        "scored": scored, "coverage": scored / eligible if eligible else 0,
        "losses_lower_is_better": {k: fmean(v) for k, v in losses.items()},
        "binary_reliability_bins": calibration_bins(observations), "failures": failures,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "model_calls": 0, "api_cost_usd": 0,
    }


def report(calibration: float = 1.0) -> dict:
    rows = load_decisions()
    inputs = [Path(__file__), ROOT / "formats.py", ROOT / "pipeline.py", ROOT / "uv.lock", DATA]
    digests = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", *digests], cwd=ROOT))
    except (FileNotFoundError, subprocess.CalledProcessError):
        head, dirty = None, None
    # Both profile timers start after the official numeric validation code is loaded.
    warmup_meeting = rows[1].published
    warmup_cutoff = warmup_meeting - timedelta(days=1)
    predict(make_question("numeric", warmup_cutoff, warmup_meeting), (rows[0].change_bp,),
            "baseline", warmup_cutoff, rows[0].published)
    return {
        "experiment": "Walk-forward FOMC reference-class comparison",
        "holdout_start": HOLDOUT.isoformat(), "last_decision": rows[-1].published.isoformat(),
        "measurement_unit": "scheduled FOMC meeting, four correlated question formats",
        "evaluated_meetings": sum(row.published >= HOLDOUT for row in rows),
        "python": platform.python_version(), "git_base": head, "inputs_differ_from_base": dirty,
        "sha256": digests,
        "profiles": {name: evaluate(rows, name, 1.0 if name == "baseline" else calibration)
                     for name in PROFILES},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation_report.json")
    parser.add_argument("--experimental-calibration", type=float, default=1.0)
    args = parser.parse_args()
    results = report(args.experimental_calibration)
    args.output.write_text(json.dumps(results, allow_nan=False, indent=2) + "\n")
    print(f"Wrote {args.output}")
    raise SystemExit(int(any(run["coverage"] < 1 for run in results["profiles"].values())))
