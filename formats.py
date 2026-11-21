"""Interpret individual FutureEval questions and validate API forecast payloads."""

from __future__ import annotations

import math
import os
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import fmean

# The SDK imports LiteLLM transitively. Its packaged model catalog suffices here.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90)
FIELDS = {
    "binary": "probability_yes",
    "multiple_choice": "probability_yes_per_category",
    "numeric": "continuous_cdf",
    "discrete": "continuous_cdf",
}


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError("expected a finite JSON number")
    return float(value)


def _probability(value: object) -> float:
    p = _number(value)
    if not 0 <= p <= 1:
        raise ValueError("expected a probability in [0, 1]")
    return p


def _time(value: str) -> datetime:
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if date.utcoffset() is None:
        raise ValueError("expected a timestamp with UTC offset")
    return date.astimezone(timezone.utc)


def _grid(q: dict, scaling: dict) -> tuple[float, ...]:
    low, high = _number(scaling["range_min"]), _number(scaling["range_max"])
    if low >= high:
        raise ValueError("unordered scaling bounds")
    raw = scaling.get("continuous_range")
    n = scaling.get("inbound_outcome_count")
    if n is None:
        n = q.get("inbound_outcome_count")
    if n is None:
        if isinstance(raw, list):
            n = len(raw) - 1
        elif q["type"] == "numeric":
            n = 200
    if type(n) is not int or not 1 <= n <= 10000:
        raise ValueError("invalid inbound outcome count")
    if raw is not None:
        if not isinstance(raw, list) or len(raw) != n + 1:
            raise ValueError("provided grid length disagrees with outcome count")
        grid = tuple(_number(x) for x in raw)
        if not (math.isclose(grid[0], low, abs_tol=1e-10) and math.isclose(grid[-1], high, abs_tol=1e-10)):
            raise ValueError("grid endpoints disagree with bounds")
    else:
        zero = scaling.get("zero_point")
        if zero is None:
            interior = [low + (high - low) * (i / n) for i in range(1, n)]
        else:
            zero = _number(zero)
            if low <= zero <= high:
                raise ValueError("log scaling zero point within bounds")
            # expm1 retains precision for nearly linear log scales.
            log_ratio = math.log1p((high - low) / (low - zero))
            interior = [low + (low - zero) * math.expm1(i * log_ratio / n) for i in range(1, n)]
        grid = (low, *interior, high)
    if any(b <= a for a, b in zip(grid, grid[1:])):
        raise ValueError("grid must be strictly increasing")
    return grid


@dataclass(frozen=True)
class Question:
    id: int
    post_id: int
    kind: str
    title: str
    close_time: datetime
    options: tuple[str, ...]
    grid: tuple[float, ...]
    open_lower_bound: bool
    open_upper_bound: bool
    metadata: dict

    @classmethod
    def from_post(cls, post: dict, *, respect_spot: bool = True) -> Question:
        q = post["question"]
        kind = q["type"]
        if kind not in FIELDS:
            raise ValueError("unsupported question type")
        if any(type(i) is not int or i <= 0 for i in (post["id"], q["id"])):
            raise ValueError("invalid post or question ID")
        if q.get("post_id", post["id"]) != post["id"]:
            raise ValueError("question belongs to another post")
        if not isinstance(q["title"], str) or not q["title"].strip():
            raise ValueError("missing question title")
        if not isinstance(q.get("resolution_criteria"), str):
            raise ValueError("resolution criteria unavailable")
        times = [_time(q["scheduled_close_time"])]
        deadlines = ("actual_close_time", "spot_scoring_time") if respect_spot else ("actual_close_time",)
        for deadline in deadlines:
            if q.get(deadline):
                times.append(_time(q[deadline]))
        metadata = {k: q.get(k) for k in (
            "title", "description", "resolution_criteria", "fine_print", "unit",
            "open_time", "scheduled_close_time", "actual_close_time", "scheduled_resolve_time",
            "group_variable", "label", "spot_scoring_time",
        )}
        options = ()
        grid = ()
        lower_open = upper_open = False
        if kind == "multiple_choice":
            values = q["options"]
            if (not isinstance(values, list) or len(values) < 2
                    or any(not isinstance(k, str) or not k.strip() for k in values)
                    or len(set(values)) != len(values)):
                raise ValueError("invalid multiple-choice option labels")
            options = tuple(values)
            metadata["options"] = list(options)
        if kind in ("numeric", "discrete"):
            scaling = q["scaling"]
            grid = _grid(q, scaling)
            lower_open = q.get("open_lower_bound", scaling.get("open_lower_bound"))
            upper_open = q.get("open_upper_bound", scaling.get("open_upper_bound"))
            if type(lower_open) is not bool or type(upper_open) is not bool:
                raise ValueError("bound openness unavailable")
            metadata["scaling"] = {k: scaling.get(k) for k in (
                "range_min", "range_max", "nominal_min", "nominal_max", "zero_point",
            )}
            metadata["scaling"]["inbound_outcome_count"] = len(grid) - 1
            metadata.update(open_lower_bound=lower_open, open_upper_bound=upper_open)
        return cls(q["id"], post["id"], kind, q["title"], min(times), options,
                   grid, lower_open, upper_open, metadata)

    def context(self) -> dict:
        """Allowlisted metadata only. Never serialize aggregates or other forecasts."""
        return {**self.metadata, "type": self.kind, "deadline": self.close_time.isoformat()}


def format_instruction(q: Question) -> str:
    if q.kind == "binary":
        return "Return the probability of Yes as a single JSON number between 0 and 1."
    if q.kind == "multiple_choice":
        return "Return a JSON object with all exact option labels as keys and probabilities summing to 1."
    return ("Return a JSON list of five nondecreasing outcome values at the 10th, 25th, "
            "50th, 75th and 90th percentiles, in the question's units. Respect closed "
            "bounds. Values may extend beyond open bounds. Repeated values are allowed.")


def _grid_coordinate(grid: tuple[float, ...], value: float) -> float:
    """Interpolate on the supplied nominal grid; extrapolate from its end intervals."""
    i = min(max(1, bisect_right(grid, value)), len(grid) - 1)
    return i - 1 + (value - grid[i - 1]) / (grid[i] - grid[i - 1])


def _cdf(q: Question, values: list) -> list[float]:
    if not isinstance(values, list) or len(values) != len(LEVELS):
        raise ValueError("expected five ordered quantile values")
    xs = [_grid_coordinate(q.grid, _number(v)) for v in values]
    n = len(q.grid) - 1
    if any(a > b for a, b in zip(xs, xs[1:])):
        raise ValueError("unordered quantiles")
    if (not q.open_lower_bound and xs[0] < 0) or (not q.open_upper_bound and xs[-1] > n):
        raise ValueError("quantile outside a closed bound")
    # Place the unobserved tails by extrapolating the nearest quantile segment.
    # Repeated quantiles contribute a point mass. At a closed lower bound it
    # belongs in the first inbound bin; at a closed upper bound, in the last.
    low = xs[0] - (xs[1] - xs[0]) * (LEVELS[0] / (LEVELS[1] - LEVELS[0]))
    high = xs[-1] + (xs[-1] - xs[-2]) * ((1 - LEVELS[-1]) / (LEVELS[-1] - LEVELS[-2]))
    xs = [low if q.open_lower_bound else max(0.0, low), *xs,
          high if q.open_upper_bound else min(n, high)]
    ps = (0.0, *LEVELS, 1.0)
    raw = []
    for edge in range(n + 1):
        i = bisect_left(xs, edge) if edge == 0 else bisect_right(xs, edge)
        if i == 0:
            p = 0.0
        elif i == len(xs):
            p = 1.0
        else:
            p = ps[i - 1] + (ps[i] - ps[i - 1]) * ((edge - xs[i - 1]) / (xs[i] - xs[i - 1]))
        raw.append(p)
    # Reserve the required uniform component and a small numerical margin.
    uniform_mass = 0.010001
    lower_tail = 0.001001 if q.open_lower_bound else 0.0
    upper_tail = 0.001001 if q.open_upper_bound else 0.0
    scale = 1.0 - lower_tail - upper_tail - uniform_mass
    cdf = [lower_tail + uniform_mass * (i / n) + scale * p for i, p in enumerate(raw)]
    start, end = cdf[0] if q.open_lower_bound else 0.0, cdf[-1] if q.open_upper_bound else 1.0
    cdf[0], cdf[-1] = start, end
    # Spread only the mass above the platform cap, proportionally to the
    # original inbound probabilities. Monotone bisection has bounded runtime.
    masses = [b - a for a, b in zip(cdf, cdf[1:])]
    cap = 39.9999 / n
    if max(masses) > cap:
        lo, hi = 1.0, cap / min(masses)
        for _ in range(60):
            mid = (lo + hi) / 2
            if math.fsum(min(cap, mid * m) for m in masses) < end - start:
                lo = mid
            else:
                hi = mid
        for i, mass in enumerate(masses[:-1], start=1):
            cdf[i] = cdf[i - 1] + min(cap, hi * mass)
    return cdf


def prediction_payload(q: Question, raw: object) -> dict:
    """Convert a model's structured estimate to a native forecast API payload."""
    if q.kind == "binary":
        value = min(0.999, max(0.001, _probability(raw)))
    elif q.kind == "multiple_choice":
        from forecasting_tools import PredictedOption, PredictedOptionList

        if not isinstance(raw, dict) or set(raw) != set(q.options):
            raise ValueError("forecast option labels do not match the question")
        probabilities = {k: _probability(raw[k]) for k in q.options}
        total = math.fsum(probabilities.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError("option probabilities too far from one")
        # Normalize rounding before calling the SDK so it never logs the raw sum.
        options = [PredictedOption(option_name=k, probability=p / total) for k, p in probabilities.items()]
        # SDK normalization enforces the lower bound and redistributes remaining mass.
        value = PredictedOptionList(predicted_options=options).to_dict()
    else:
        value = _cdf(q, raw)
    payload = {FIELDS[q.kind]: value}
    validate_payload(q, payload)
    return payload


def validate_payload(q: Question, payload: dict) -> None:
    key = FIELDS[q.kind]
    if not isinstance(payload, dict) or set(payload) != {key}:
        raise ValueError("unexpected forecast fields")
    value = payload[key]
    if q.kind == "binary":
        if not 0.001 <= _probability(value) <= 0.999:
            raise ValueError("binary forecast violates platform bounds")
    elif q.kind == "multiple_choice":
        if not isinstance(value, dict) or set(value) != set(q.options):
            raise ValueError("forecast option labels do not match the question")
        if any(not 0.001 <= _probability(p) <= 0.999 for p in value.values()):
            raise ValueError("option probability outside platform bounds")
        if abs(math.fsum(value.values()) - 1.0) > 1e-9:
            raise ValueError("option probabilities must sum to one")
    else:
        if not isinstance(value, list) or len(value) != len(q.grid):
            raise ValueError("wrong CDF length")
        for p in value:
            _probability(p)
        if (value[0] < 0.001 if q.open_lower_bound else value[0] != 0.0):
            raise ValueError("wrong lower tail")
        if (value[-1] > 0.999 if q.open_upper_bound else value[-1] != 1.0):
            raise ValueError("wrong upper tail")
        n = len(value) - 1
        for a, b in zip(value, value[1:]):
            if not 0.01 / n - 1e-12 <= b - a <= 40.0 / n + 1e-12:
                raise ValueError("CDF increment outside platform bounds")


def aggregate(q: Question, payloads: list[dict], calibration: float = 1.0) -> dict:
    """Linear opinion pool with optional binary log-odds scaling."""
    if not payloads or _number(calibration) <= 0:
        raise ValueError("positive calibration and at least one forecast required")
    for payload in payloads:
        validate_payload(q, payload)
    key = FIELDS[q.kind]
    values = [p[key] for p in payloads]
    if q.kind == "binary":
        p = fmean(values)
        if calibration != 1:
            z = calibration * math.log(p / (1 - p))
            exp_z = math.exp(-abs(z))
            p = 1 / (1 + exp_z) if z >= 0 else exp_z / (1 + exp_z)
        return prediction_payload(q, p)
    if q.kind == "multiple_choice":
        pooled = {k: fmean(v[k] for v in values) for k in q.options}
        return prediction_payload(q, pooled)
    else:
        pooled = [fmean(column) for column in zip(*values)]
    result = {key: pooled}
    validate_payload(q, result)
    return result


def display_forecast(q: Question, payload: dict) -> str:
    validate_payload(q, payload)
    value = payload[FIELDS[q.kind]]
    if q.kind == "binary":
        return f"P(Yes) = {value:.4%}"
    if q.kind == "multiple_choice":
        return ", ".join(f"{k} = {value[k]:.4%}" for k in q.options)
    descriptions = []
    for level in LEVELS:
        i = bisect_left(value, level)
        if i == 0:
            estimate = f"below {q.grid[0]:g}"
        elif i == len(value):
            estimate = f"above {q.grid[-1]:g}"
        else:
            x, y = q.grid[i - 1], q.grid[i]
            fraction = 0.5 if q.kind == "discrete" else (level - value[i - 1]) / (value[i] - value[i - 1])
            estimate = f"{x + (y - x) * fraction:.6g}"
        descriptions.append(f"P{round(100 * level)} = {estimate}")
    return ", ".join(descriptions) + f" ({q.metadata.get('unit') or 'question units'})"
