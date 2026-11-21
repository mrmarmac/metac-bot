"""Opt-in prospective comparison on an independently defined USGS catalog event."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
from statistics import fmean
import subprocess
import tempfile
import time

import requests

from evaluation import score_outcome
from formats import Question
from pipeline import (BASELINE, Evidence, ProviderError, complete, configured_key,
                      forecast_with_inputs)
from state import BudgetExceeded, GitHubFile, Ledger


WEEK = timedelta(days=7)
CATALOG = "https://earthquake.usgs.gov/fdsnws/event/1/query"
ROOT = Path(__file__).resolve().parent


def utcnow():
    return datetime.now(timezone.utc)


def next_window(now: datetime) -> datetime:
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight + timedelta(days=(7 - midnight.weekday()) or 7)


def sample_catalog(start: datetime, end: datetime) -> tuple[str, list[tuple[str, int]]]:
    """Use the complete USGS result. Fail on truncation, duplicates or an invalid response."""
    response = requests.get(CATALOG, params={"format": "geojson", "minmagnitude": 6,
                            "eventtype": "earthquake", "starttime": start.isoformat(),
                            "endtime": end.isoformat()}, timeout=(4, 12), allow_redirects=False)
    if response.status_code == 204:
        return response.url, []
    if response.status_code != 200 or len(response.content) > 1_000_000:
        raise RuntimeError("USGS snapshot unavailable")
    data = response.json()
    features = data["features"]
    if (data.get("type") != "FeatureCollection" or not isinstance(features, list)
            or len(features) != data["metadata"]["count"] or len(features) >= 20000):
        raise ValueError("incomplete USGS catalog")
    rows = []
    for feature in features:
        stamp, ident = feature["properties"]["time"], feature["id"]
        if type(stamp) is not int or not isinstance(ident, str):
            raise ValueError("malformed USGS event")
        if start.timestamp() * 1000 <= stamp < end.timestamp() * 1000:
            rows.append((ident, stamp))
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError("duplicate USGS event")
    return response.url, rows


def question_post(start: datetime) -> dict:
    end = start + WEEK
    ident = start.date().toordinal()
    return {"id": ident, "question": {"id": ident, "title": "Will the USGS magnitude-6 catalog "
            f"query return at least three global earthquakes in the UTC week starting {start.date()}?",
            "type": "binary", "description": "An independent prospective research question.",
            "resolution_criteria": f"Count unique events returned by the worldwide USGS event query "
            f"with minmagnitude=6 and eventtype=earthquake, origin times in [{start.isoformat()}, "
            f"{end.isoformat()}). Use the first saved catalog snapshot from 24 hours through 7 days "
            "after the window ends. Resolve Yes if the count is at least 3.",
            "fine_print": "Later catalog revisions do not change the recorded outcome.",
            "scheduled_close_time": start.isoformat(), "scheduled_resolve_time": (end + timedelta(days=1)).isoformat()}}


def capture(folder: Path, ledger: Ledger, key: str) -> dict:
    begin = utcnow()
    start = next_window(begin)
    if (start - begin).total_seconds() < 240:
        raise ValueError("insufficient time to freeze both forecasts")
    # End the reference periods at the start of today's UTC day, before fetching evidence.
    cutoff = begin.replace(hour=0, minute=0, second=0, microsecond=0)
    since = cutoff - 8 * WEEK
    url, observations = sample_catalog(since, cutoff)
    counts = [0] * 8
    for _, ms in observations:
        index = (datetime.fromtimestamp(ms / 1000, timezone.utc) - since) // WEEK
        counts[index] += 1
    now = utcnow()
    text = f"Counts in eight successive seven-day periods ending {cutoff.date()} UTC: {counts}."
    evidence = Evidence(1, now.isoformat(), url, "USGS earthquake catalog", text)
    post = question_post(start)
    q = Question.from_post(post)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{start.date()}.capture.json"
    # Claim before incurring costs. A previous attempt is never silently rerun.
    with path.open("x", encoding="utf8") as out:
        def claim(data):
            trials = data.setdefault("forward_trials", {})
            trial = start.date().isoformat()
            if trial in trials:
                raise ValueError("prospective trial already claimed")
            trials[trial] = now.isoformat()

        ledger.change(claim)
        members = {}
        def infer(member, messages, deadline):
            stamp = time.monotonic()
            prior_ids = set(ledger.data["calls"])
            try:
                answer = complete(member, messages, -q.id, ledger, key, deadline)
                members[member.model] = {"answer": answer}
                return answer
            except (ProviderError, BudgetExceeded) as exc:
                members[member.model] = {"error": type(exc).__name__}
                raise
            finally:
                if member.model in members:
                    record = members[member.model]
                    record["seconds"] = round(time.monotonic() - stamp, 3)
                    record["billing_calls"] = [v for k, v in ledger.data["calls"].items()
                                               if k not in prior_ids and v["qid"] == -q.id]

        note = "Primary USGS catalog snapshot; eight previous nonoverlapping periods."
        calibration = float(os.getenv("FORECAST_CALIBRATION", "1"))
        profiles = {}
        deadline = time.monotonic() + min(180, (start - now).total_seconds() - 10)
        try:
            # Both profiles share the observed baseline call. It is charged exactly once.
            profiles["experimental"] = forecast_with_inputs(q, deadline, "experimental", now, [evidence],
                                                             note, calibration, infer)[0]
        except (ProviderError, BudgetExceeded, ValueError):
            pass
        baseline = members.get(BASELINE.model, {}).get("answer")
        if baseline is not None:
            try:
                profiles["baseline"] = forecast_with_inputs(q, time.monotonic() + 5, "baseline", now,
                                                             [evidence], note, calibration,
                                                             lambda *args: baseline)[0]
            except (ProviderError, ValueError):
                pass
        frozen = utcnow()
        if frozen >= start:
            raise RuntimeError("capture missed immutable information cutoff")
        calls = [v for v in ledger.data["calls"].values() if v["qid"] == -q.id]
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        entry = {"version": 1, "revision": revision, "source_sha256": {
                    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in ("pipeline.py", "formats.py", "forward_eval.py", "evaluation.py", "uv.lock")},
                 "start": start.isoformat(), "post": post, "captured_at": frozen.isoformat(),
                 "evidence": asdict(evidence), "observations": observations, "members": members,
                 "profiles": profiles, "calibration": calibration, "billing_calls": calls}
        # Keep a valid capture or its empty attempt marker across interrupted writes.
        with tempfile.NamedTemporaryFile("w", dir=folder, delete=False) as temp:
            json.dump(entry, temp, allow_nan=False, indent=2)
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp.name, path)
    return {"trial": start.date().isoformat(), "profiles_captured": sorted(profiles)}


def resolve(capture_path: Path) -> dict:
    raw = capture_path.read_bytes()
    entry = json.loads(raw)
    start = datetime.fromisoformat(entry["start"])
    end = start + WEEK
    if datetime.fromisoformat(entry["captured_at"]) >= start or entry["post"] != question_post(start):
        raise ValueError("invalid prospective capture")
    path = capture_path.with_suffix(".resolved.json")
    digest = hashlib.sha256(raw).hexdigest()
    def score_snapshot(resolved):
        resolved["outcome"] = len(resolved["observations"]) >= 3
        q = Question.from_post(entry["post"])
        resolved["scores"] = {k: score_outcome(q, v, resolved["outcome"])
                              for k, v in entry["profiles"].items()}
        return resolved

    if path.exists():
        resolved = json.loads(path.read_text())
        if resolved["capture_sha256"] != digest:
            raise ValueError("capture changed after resolution")
        return score_snapshot(resolved)
    now = utcnow()
    if not end + timedelta(days=1) <= now < end + WEEK:
        raise ValueError("outside prespecified resolution window")
    url, observations = sample_catalog(start, end)
    recorded_at = utcnow()
    if recorded_at >= end + WEEK:
        raise ValueError("resolution request missed the cutoff")
    resolved = score_snapshot({"capture_sha256": digest, "snapshot_at": recorded_at.isoformat(),
                               "url": url, "observations": observations})
    with path.open("x", encoding="utf8") as out:
        json.dump(resolved, out, indent=2)
    return resolved


def report(folder: Path) -> dict:
    trials = sorted(folder.glob("*.capture.json"))
    rows = []
    operations = {p: {"coverage_count": 0, "member_seconds": 0., "settled_usd": 0.,
                      "reserved_usd": 0.} for p in ("baseline", "experimental")}
    for trial in trials:
        # A crash can leave an empty create-only claim; retain it in the denominator.
        if trial.stat().st_size == 0:
            continue
        entry = json.loads(trial.read_text())
        for profile, stats in operations.items():
            stats["coverage_count"] += int(profile in entry["profiles"])
            for model, member in entry["members"].items():
                if profile == "baseline" and model != BASELINE.model:
                    continue
                stats["member_seconds"] += member["seconds"]
                for call in member["billing_calls"]:
                    field = "settled_usd" if call["settled"] else "reserved_usd"
                    stats[field] += call["amount"] / 1e6
        path = trial.with_suffix(".resolved.json")
        if path.is_file():
            rows.append(resolve(trial))
    paired = [r for r in rows if set(r["scores"]) == {"baseline", "experimental"}]
    return {"attempted": len(trials), "resolved": len(rows), "paired": len(paired),
            "operations": operations, "paired_mean_scores": {
                profile: {metric: fmean(r["scores"][profile][metric] for r in paired)
                          for metric in ("binary_brier", "binary_log_loss")}
                for profile in ("baseline", "experimental")} if paired else {}}



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    capture_args = sub.add_parser("capture")
    capture_args.add_argument("folder", type=Path)
    capture_args.add_argument("--spend", action="store_true", required=True)
    sub.add_parser("resolve").add_argument("file", type=Path)
    sub.add_parser("report").add_argument("folder", type=Path)
    args = parser.parse_args()
    if args.mode == "capture":
        key = configured_key()
        ledger = Ledger(GitHubFile(), os.getenv("PER_QUESTION_USD") or "0", os.getenv("MONTHLY_USD") or "0")
        result = capture(args.folder, ledger, key)
    elif args.mode == "resolve":
        result = resolve(args.file)["scores"]
    else:
        result = report(args.folder)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
