"""
Assignment 2 cost sweep: what each pair of thresholds costs per 1,000 bags.

    uv run python costing.py     # reads artifacts/val_predictions.json and artifacts/test_predictions.json
                                 # -> artifacts/cost_sweep.json, artifacts/cost_curve.png

Also run at the end of train.py, and imported by the dashboard (app.py).

Method:

- The decision unit is one bag, i.e. one image.
- Thresholds are CHOSEN on the validation split (artifacts/val_predictions.json,
  312 images) and their cost is REPORTED on the test split (156 images), so the
  test number is not flattered by picking the threshold on it. The choice is
  made on a grid of 2-decimal thresholds (0.01 to 0.99).
- A missed threat costs what the threat is: each class has its own miss cost
  (MISS_COST below), and a threat bag's miss cost is that of the most severe
  object labelled in it (gun > knife > scissors > pliers / wrench).
- The splits hold far more guns than real traffic would, so each threat bag is
  weighted by an assumed mix of threat bags by most severe class (THREAT_MIX):
  a bag of class c weighs THREAT_MIX[c] / (number of threat bags of class c in
  the split). Every mean over threat bags (cost, shares, per-class splits) uses
  these weights. Counts ("counts") are plain bag counts.
- Threat-bag score: the highest confidence among ALL boxes in the image (any
  class, wherever they are). The decision is whether to open the bag, so if the
  model fires anything on a bag that holds a threat, the bag is searched and the
  threat found. A threat is missed only when no box reaches the threshold.
- The splits have no clean bags, so a clean bag is approximated by the empty
  content of each image: the highest confidence among the boxes that do not
  overlap any real object (IoU <= CLEAN_MAX_IOU with every ground-truth box).
  It is only an approximation: the empty part of a threat bag is smaller and
  more cluttered than a real clean bag.
- Model alone: a bag is searched if its score >= t, otherwise it passes.
- Model + person (the chosen policy: the cost-optimal pair t_low / t_high):
  score < t_low passes, t_low <= score < t_high goes to a screener for a second
  look, score >= t_high goes straight to a manual search.
  The screener catches a reviewed threat with probability REVIEWER_CATCH (then
  it is searched) and clears a reviewed clean bag: by assumption a screener
  never sends a clean bag on to a search (SCREENER_SENDS_CLEAN = 0; the
  sensitivity table shows 10%).
- Expected cost per 1,000 bags =
      1000 x [ p x (weighted mean threat bag cost) + (1 - p) x (clean bag cost) ]
  where p is the base rate.
"""

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
PREDICTIONS = ARTIFACTS_DIR / "test_predictions.json"  # where costs are reported
VAL_PREDICTIONS = ARTIFACTS_DIR / "val_predictions.json"  # where thresholds are chosen
MODEL = "finetuned"  # key of the finetuned model in metrics.json and *_predictions.json


def model_key(keys, model: str = MODEL) -> str:
    """The finetuned model's key: `model` if present, else the first key starting with "finetuned"."""
    keys = list(keys)
    if model in keys:
        return model
    for key in keys:
        if key.startswith("finetuned"):
            return key
    raise KeyError(f"no finetuned model among {keys}")


# ------------------------------------------------------------------------- costs
# Screener time and pay
SCREENER_EUR_PER_HOUR = 20.0  # loaded screener cost
SEARCH_MINUTES = 2  # officer time per manual search
SEARCH_OFFICER_ONLY = SEARCH_MINUTES / 60 * SCREENER_EUR_PER_HOUR  # EUR 0.67, sensitivity case

# A manual search costs EUR 2: the officer's time (2 min at EUR 20/h = EUR 0.67)
# plus the passenger's wait and the lane slowing down while the bag is opened.
C_FP = 2.0  # clean bag searched for nothing (EUR)
C_TP = 2.0  # threat found: the same manual search (EUR)
C_TN = 0.0  # clean bag passes (EUR)

# Cost of a threat that gets through, by what it is (EUR). Round figures set by
# how much harm the item can do on board: a gun is the worst case, a knife half
# of it, scissors a tenth of a knife, pliers and a wrench lower again (blunt
# tools that are hard to use as a weapon in a cabin).
MISS_COST = {
    "Gun": 1_000_000.0,
    "Knife": 500_000.0,
    "Scissors": 50_000.0,
    "Pliers": 20_000.0,
    "Wrench": 20_000.0,
}

# ASSUMPTION: what the threat bags at a real checkpoint hold, by most severe
# object. Round, plausible shares, not from a source: guns are rare, small
# scissors are the most common prohibited item, then knives and tools. The test
# split instead has a gun in 49 of 156 threat bags (31%), which would mean about
# 28 guns a day at 90,700 bags.
THREAT_MIX = {"Gun": 0.01, "Knife": 0.20, "Scissors": 0.45, "Pliers": 0.17, "Wrench": 0.17}

# Base rate: most bags hold nothing prohibited.
BAGS_PER_THREAT = 1_000  # 1 bag in 1,000 carries a prohibited item, the other 999 are clean
BASE_RATE = 1 / BAGS_PER_THREAT

# Review load
REVIEW_SECONDS = 30  # a second look at one image
REVIEW_SHARE_OF_TIME = 20 / 30  # 20 min of image review, then a 10 min break
SHIFT_HOURS = 8  # -> 8 x 20/30 = 5.3 hours of image review per shift
BAGS_PER_DAY = 90_700  # Madrid-Barajas departing bags a day
# Paid cost of one review, breaks included: 30 s x EUR 20/h / (20/30) = EUR 0.25.
REVIEW_COST = REVIEW_SECONDS / 3600 * SCREENER_EUR_PER_HOUR / REVIEW_SHARE_OF_TIME

# A screener who checks an image catches 98% of the threats in the bags they review.
REVIEWER_CATCH = 0.98
# ASSUMPTION: a screener never sends a clean bag on to a manual search (no false alarms).
SCREENER_SENDS_CLEAN = 0.0

# ------------------------------------------------------------------ sweep set-up
CLEAN_MAX_IOU = 0.1  # a box counts as "on empty content" if its IoU with every real object is <= this
GRID = np.round(np.arange(1, 100) / 100, 2)  # 0.01 to 0.99: thresholds are chosen on this grid
THRESHOLDS = np.round(np.arange(1, 991) / 1000, 3)  # 0.001 to 0.990: fine grid for the curves only
T_HIGH = 0.7  # the earlier proposed upper threshold, kept as an alternative to the cost-optimal pair


@dataclass(frozen=True)
class Costs:
    """Everything the expected cost depends on. Change one field with dataclasses.replace."""

    fp: float = C_FP
    tp: float = C_TP
    tn: float = C_TN
    # Missed threat, by the most severe object in the bag
    fn_gun: float = MISS_COST["Gun"]
    fn_knife: float = MISS_COST["Knife"]
    fn_scissors: float = MISS_COST["Scissors"]
    fn_pliers: float = MISS_COST["Pliers"]
    fn_wrench: float = MISS_COST["Wrench"]
    base_rate: float = BASE_RATE
    review: float = REVIEW_COST
    reviewer_catch: float = REVIEWER_CATCH
    reviewer_sends_clean: float = SCREENER_SENDS_CLEAN  # share of reviewed clean bags sent on to a search

    def miss_cost(self, name: str) -> float:
        """Cost of missing a threat bag whose most severe object is class `name`."""
        return getattr(self, f"fn_{name.lower()}")


DEFAULT_COSTS = Costs()
MISS_FIELDS = tuple(f"fn_{name.lower()}" for name in MISS_COST)
# Most severe first. Fixed, so a sensitivity change to one cost never regroups the bags.
SEVERITY = sorted(MISS_COST, key=MISS_COST.get, reverse=True)


def scale_miss(costs: Costs, factor: float) -> Costs:
    """Every class's miss cost multiplied by factor."""
    return replace(costs, **{f: getattr(costs, f) * factor for f in MISS_FIELDS})


def flat_miss(costs: Costs, value: float) -> Costs:
    """The same miss cost for every class."""
    return replace(costs, **{f: value for f in MISS_FIELDS})


@dataclass(frozen=True)
class BagScores:
    """One score per bag of a split: threat = boxes anywhere, clean = boxes on background.

    threat_class: the most severe ground-truth class in each threat bag (sets its miss cost).
    threat_weight: each threat bag's weight in the mean over threat bags (sums to 1).
    """

    threat: np.ndarray
    clean: np.ndarray
    threat_class: np.ndarray
    threat_weight: np.ndarray


# ------------------------------------------------------------------- bag scores
def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(len(a), len(b)) IoU between x1, y1, x2, y2 boxes."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def image_scores(gt: np.ndarray, pred: np.ndarray, clean_max_iou: float = CLEAN_MAX_IOU) -> tuple[float, float]:
    """(threat-bag score, clean stand-in score) for one image. gt: class,x1,y1,x2,y2. pred: class,conf,x1,y1,x2,y2."""
    if not len(pred):
        return 0.0, 0.0
    threat = float(pred[:, 1].max())
    overlap = iou(pred[:, 2:6], gt[:, 1:5]).max(axis=1) if len(gt) else np.zeros(len(pred))
    empty = overlap <= clean_max_iou
    clean = float(pred[empty, 1].max()) if empty.any() else 0.0
    return threat, clean


def most_severe(gt: np.ndarray, names: list[str]) -> str:
    """Name of the most severe ground-truth class in one image (gt: class,x1,y1,x2,y2)."""
    present = {names[int(c)] for c in gt[:, 0]}
    if not present:
        raise ValueError("an image has no ground-truth object, so it is not a threat bag")
    return min(present, key=SEVERITY.index)


def mix_weights(threat_class: np.ndarray, mix: dict | None = THREAT_MIX) -> np.ndarray:
    """Per-bag weight: mix[c] / (bags of class c), renormalised over the classes present.

    mix=None: every bag weighs the same (the split's own mix).
    """
    if mix is None:
        return np.full(len(threat_class), 1 / len(threat_class))
    counts = {c: int((threat_class == c).sum()) for c in set(threat_class)}
    w = np.array([mix[c] / counts[c] for c in threat_class])
    return w / w.sum()


def load_bag_scores(path: Path = PREDICTIONS, model: str | None = None, clean_max_iou: float = CLEAN_MAX_IOU,
                    mix: dict | None = THREAT_MIX) -> BagScores:
    data = json.loads(Path(path).read_text())
    names, images = data["names"], data["images"]
    key = model or model_key(k for k in images[0] if k not in ("file", "gt"))
    gts = [np.array(im["gt"]).reshape(-1, 5) for im in images]
    pairs = [image_scores(gt, np.array(im[key]).reshape(-1, 6), clean_max_iou) for gt, im in zip(gts, images)]
    threat, clean = np.array(pairs).T
    cls = np.array([most_severe(gt, names) for gt in gts])
    return BagScores(threat=threat, clean=clean, threat_class=cls, threat_weight=mix_weights(cls, mix))


def load_splits(clean_max_iou: float = CLEAN_MAX_IOU, mix: dict | None = THREAT_MIX) -> tuple[BagScores, BagScores]:
    """(validation, test) bag scores."""
    return (load_bag_scores(VAL_PREDICTIONS, clean_max_iou=clean_max_iou, mix=mix),
            load_bag_scores(PREDICTIONS, clean_max_iou=clean_max_iou, mix=mix))


# --------------------------------------------------------------------- outcomes
def share_at_or_above(scores: np.ndarray, t, weights: np.ndarray | None = None) -> np.ndarray:
    """(Weighted) share of scores >= t, for a scalar or an array of thresholds."""
    order = np.argsort(scores, kind="stable")
    w = np.ones(len(scores)) if weights is None else np.asarray(weights, dtype=float)[order]
    cum = np.concatenate([[0.0], np.cumsum(w)])
    below = np.searchsorted(scores[order], np.asarray(t, dtype=float), side="left")
    return 1 - cum[below] / cum[-1]


def _split(s: np.ndarray, t_low, t_high, w: np.ndarray | None = None) -> dict:
    above_low, above_high = share_at_or_above(s, t_low, w), share_at_or_above(s, t_high, w)
    return {"pass": 1 - above_low, "review": above_low - above_high, "search": above_high}


def outcomes(scores: BagScores, t_low, t_high=None) -> dict:
    """Weighted share of threat bags and share of clean bags that pass, go to review, or go to search.

    t_high=None means the model alone (t_high = t_low, no review band).
    Works on scalars or broadcastable arrays of thresholds (+/- inf sends every bag one way).
    "threat_by_class" splits the threat bags by their most severe object: the same three
    shares within each group, plus the group's weight (its share of threat bags, i.e. THREAT_MIX).
    """
    t_high = t_low if t_high is None else t_high
    out = {"threat": _split(scores.threat, t_low, t_high, scores.threat_weight),
           "clean": _split(scores.clean, t_low, t_high)}
    out["threat_by_class"] = {}
    for name in SEVERITY:
        g = scores.threat_class == name
        if g.any():
            w = scores.threat_weight[g]
            out["threat_by_class"][name] = {"weight": float(w.sum()), **_split(scores.threat[g], t_low, t_high, w)}
    return out


def cost_per_1000(o: dict, costs: Costs = DEFAULT_COSTS):
    """Expected EUR per 1,000 bags for outcome shares from outcomes()."""
    c, k = o["clean"], costs
    threat_bag = 0.0  # weighted mean over threat bags, each with its own miss cost
    for name, g in o["threat_by_class"].items():
        fn = k.miss_cost(name)
        reviewed_threat = k.review + k.reviewer_catch * k.tp + (1 - k.reviewer_catch) * fn
        threat_bag = threat_bag + g["weight"] * (g["pass"] * fn + g["review"] * reviewed_threat + g["search"] * k.tp)
    reviewed_clean = k.review + k.reviewer_sends_clean * k.fp
    clean_bag = c["pass"] * k.tn + c["review"] * reviewed_clean + c["search"] * k.fp
    return 1000 * (k.base_rate * threat_bag + (1 - k.base_rate) * clean_bag)


def counts(scores: BagScores, t_low: float, t_high: float) -> dict:
    """Plain bag counts in the split: threat bags and clean stand-ins per outcome, and threat bags by class."""
    def split(s):
        n_pass, n_search = int((s < t_low).sum()), int((s >= t_high).sum())
        return {"pass": n_pass, "review": len(s) - n_pass - n_search, "search": n_search}

    return {
        "threat": split(scores.threat),
        "clean": split(scores.clean),
        "threat_by_class": {name: split(scores.threat[scores.threat_class == name])
                            for name in SEVERITY if (scores.threat_class == name).any()},
    }


def staffing(share: float, bags_per_day: int = BAGS_PER_DAY) -> dict:
    """Reviews a day, image-review hours a day and 8-hour screener shifts (20-minute rule)."""
    reviews = share * bags_per_day
    hours = reviews * REVIEW_SECONDS / 3600
    return {
        "reviews_per_day": reviews,
        "review_hours_per_day": hours,
        "shifts_per_day": hours / (SHIFT_HOURS * REVIEW_SHARE_OF_TIME),
    }


def search_staffing(share: float, bags_per_day: int = BAGS_PER_DAY) -> dict:
    """Searches a day (sent straight to search by the model) and 8-hour shifts to do them."""
    searches = share * bags_per_day
    return {"searches_per_day": searches, "search_shifts_per_day": searches * SEARCH_MINUTES / 60 / SHIFT_HOURS}


def evaluate(scores: BagScores, t_low: float, t_high: float | None = None, costs: Costs = DEFAULT_COSTS) -> dict:
    """Everything the dashboard shows for one pair of thresholds (t_high=None: model alone)."""
    t_high = t_low if t_high is None else t_high
    o = outcomes(scores, t_low, t_high)
    p = costs.base_rate
    t, c = o["threat"], o["clean"]
    share = float(p * t["review"] + (1 - p) * c["review"])
    search = float(p * t["search"] + (1 - p) * c["search"])
    return {
        "t_low": float(t_low),
        "t_high": float(t_high),
        "cost_per_1000": float(cost_per_1000(o, costs)),
        "review_share": share,
        "search_share": search,
        "pass_share": 1 - share - search,
        # Expected bags per 1,000 in each cell (rows: truth, columns: what happens)
        "per_1000_bags": {
            "threat": {k: float(1000 * p * v) for k, v in t.items()},
            "clean": {k: float(1000 * (1 - p) * v) for k, v in c.items()},
        },
        # Weighted share of threat bags missed (passed, or reviewed and not caught)
        "threat_missed_share": float(t["pass"] + t["review"] * (1 - costs.reviewer_catch)),
        # Plain counts in the split these scores come from
        "counts": counts(scores, t_low, t_high),
        **staffing(share),
        **search_staffing(search),
    }


# ------------------------------------------------------------------------ sweeps
def sweep(scores: BagScores, costs: Costs = DEFAULT_COSTS, thresholds: np.ndarray = THRESHOLDS) -> np.ndarray:
    """Model-alone cost per 1,000 bags at every threshold."""
    return cost_per_1000(outcomes(scores, thresholds), costs)


def policy_curve(scores: BagScores, t_high: float, costs: Costs = DEFAULT_COSTS,
                 thresholds: np.ndarray = THRESHOLDS) -> np.ndarray:
    """Model + screener cost per 1,000 bags as t_low varies, t_high fixed (NaN where t_low > t_high)."""
    cost = cost_per_1000(outcomes(scores, thresholds, np.full(len(thresholds), t_high)), costs)
    return np.where(thresholds <= t_high, cost, np.nan)


def best_threshold(scores: BagScores, costs: Costs = DEFAULT_COSTS, thresholds: np.ndarray = GRID) -> float:
    """Model-alone threshold with the lowest cost. On a tie, the lowest threshold (the safer side)."""
    return float(thresholds[np.argmin(sweep(scores, costs, thresholds))])


def pair_grid(scores: BagScores, costs: Costs = DEFAULT_COSTS, thresholds: np.ndarray = GRID) -> np.ndarray:
    """Cost of every (t_low, t_high) pair on the grid: rows t_low, columns t_high, inf where t_low > t_high."""
    lo, hi = thresholds[:, None], thresholds[None, :]
    return np.where(lo <= hi, cost_per_1000(outcomes(scores, lo, hi), costs), np.inf)


def best_pair(
    scores: BagScores,
    costs: Costs = DEFAULT_COSTS,
    thresholds: np.ndarray = GRID,
    t_high: float | None = None,
) -> tuple[float, float]:
    """(t_low, t_high) with the lowest cost, t_low <= t_high. Pass t_high to fix it and search t_low only.

    On a tie, the lowest t_low, then the lowest t_high.
    """
    if t_high is not None:
        cost = np.where(thresholds <= t_high, cost_per_1000(outcomes(scores, thresholds, t_high), costs), np.inf)
        return float(thresholds[np.argmin(cost)]), float(t_high)
    i, j = np.unravel_index(np.argmin(pair_grid(scores, costs, thresholds)), (len(thresholds),) * 2)
    return float(thresholds[i]), float(thresholds[j])


def chosen_pair(val: BagScores, costs: Costs = DEFAULT_COSTS) -> tuple[float, float]:
    """The chosen policy: the cost-optimal 2-decimal (t_low, t_high) pair on the validation split."""
    return best_pair(val, costs)


def pair_at_0_7(val: BagScores, costs: Costs = DEFAULT_COSTS) -> tuple[float, float]:
    """The earlier proposal: t_high fixed at T_HIGH, t_low the cost-optimal lower threshold for it (on val)."""
    return best_pair(val, costs, t_high=T_HIGH)


def plateau(val: BagScores, costs: Costs = DEFAULT_COSTS, rel: float = 0.0) -> dict:
    """Pairs whose validation cost is within `rel` of the optimum (rel=0: exact ties).

    Ranges of t_low and t_high over all such pairs, and along each threshold with the other at its optimum.
    """
    grid = pair_grid(val, costs)
    best = grid.min()
    near = grid <= best * (1 + rel) + 1e-9
    i, j = np.unravel_index(np.argmin(grid), grid.shape)
    lows, highs = GRID[near.any(axis=1)], GRID[near.any(axis=0)]
    lows_at, highs_at = GRID[near[:, j]], GRID[near[i, :]]
    return {
        "within": rel,
        "val_cost_per_1000": float(best),
        "pairs": int(near.sum()),
        "t_low_range": [float(lows.min()), float(lows.max())],
        "t_high_range": [float(highs.min()), float(highs.max())],
        "t_low_range_at_chosen_t_high": [float(lows_at.min()), float(lows_at.max())],
        "t_high_range_at_chosen_t_low": [float(highs_at.min()), float(highs_at.max())],
    }


def policies(val: BagScores, test: BagScores, costs: Costs = DEFAULT_COSTS) -> dict:
    """Thresholds chosen on val, each policy evaluated on val and on test."""
    t = best_threshold(val, costs)
    lo, hi = chosen_pair(val, costs)
    lo07, _ = pair_at_0_7(val, costs)
    pairs = {"model_alone": (t, t), "alternative_0_7": (lo07, T_HIGH), "chosen_policy": (lo, hi)}
    return {name: {"val": evaluate(val, a, b, costs), "test": evaluate(test, a, b, costs)}
            for name, (a, b) in pairs.items()}


def _changed(costs: Costs, field: str, value: float) -> Costs:
    return scale_miss(costs, value) if field == "miss_scale" else replace(costs, **{field: value})


def t_low_moves(val: BagScores, field: str, start: float, end: float, costs: Costs = DEFAULT_COSTS) -> dict | None:
    """Value of `field` between start (today) and end at which the chosen t_low first changes.

    field: "miss_scale" (a factor on every class's miss cost) or a Costs field such as "base_rate".
    Bisection on a log scale. None if t_low is the same all the way to `end`.
    """
    base = chosen_pair(val, costs)[0]
    if chosen_pair(val, _changed(costs, field, end))[0] == base:
        return None
    same, moved = start, end
    for _ in range(50):
        mid = float(np.sqrt(same * moved))
        if chosen_pair(val, _changed(costs, field, mid))[0] == base:
            same = mid
        else:
            moved = mid
    return {"value": moved, "t_low_after": chosen_pair(val, _changed(costs, field, moved))[0],
            "t_high_after": chosen_pair(val, _changed(costs, field, moved))[1]}


def reviewer_break_even(scores: BagScores, pair: tuple[float, float], t_alone: float,
                        costs: Costs = DEFAULT_COSTS) -> float:
    """Screener catch rate below which `pair` costs more than the model alone at t_alone (same split).

    The policy's cost is a straight line in the catch rate, so two points give it.
    """
    target = evaluate(scores, t_alone, None, costs)["cost_per_1000"]
    c0, c1 = (evaluate(scores, *pair, replace(costs, reviewer_catch=r))["cost_per_1000"] for r in (0.0, 1.0))
    return float((c0 - target) / (c0 - c1)) if c0 != c1 else float("nan")


SENSITIVITY_CASES = (
    [("baseline", {})]
    + [(f"missed gun EUR {v:,.0f}", {"fn_gun": v}) for v in (500_000, 2_000_000)]
    + [(f"missed knife EUR {v:,.0f}", {"fn_knife": v}) for v in (250_000, 1_000_000)]
    + [("every missed threat EUR 1,000,000", {"flat_miss": 1_000_000})]
    + [(f"base rate 1 in {n:,}", {"base_rate": 1 / n}) for n in (500, 5_000)]
    + [(f"screener catches {100 * v:g}%", {"reviewer_catch": v}) for v in (0.9, 0.95, 0.99, 1.0)]
    + [(f"manual search EUR {SEARCH_OFFICER_ONLY:.2f} (officer time only)",
        {"fp": SEARCH_OFFICER_ONLY, "tp": SEARCH_OFFICER_ONLY})]
    + [("screener sends 10% of checked clean bags to search", {"reviewer_sends_clean": 0.1})]
    + [("threat mix as labelled in the splits", {"mix": None})]
    # The clean stand-in rule changes the scores themselves, not the costs
    + [(f"clean stand-in IoU <= {v:g}", {"clean_max_iou": v}) for v in (0.0, 0.3)]
)


def sensitivity(val: BagScores, test: BagScores, costs: Costs = DEFAULT_COSTS) -> list[dict]:
    """Thresholds re-chosen on val and costed on test when one assumption changes (one row per change)."""
    rows = []
    for label, change in SENSITIVITY_CASES:
        v, t, k = val, test, costs
        if "clean_max_iou" in change:
            v, t = load_splits(clean_max_iou=change["clean_max_iou"])
        elif "mix" in change:
            v, t = load_splits(mix=change["mix"])
        elif "flat_miss" in change:
            k = flat_miss(costs, change["flat_miss"])
        else:
            k = replace(costs, **change)
        p = policies(v, t, k)
        rows.append({
            "case": label,
            "change": change,
            **{f"{name}_{key}": p[name]["test"][key] for name in p for key in ("t_low", "t_high")},
            **{f"{name}_{split}_cost_per_1000": p[name][split]["cost_per_1000"] for name in p for split in ("val", "test")},
            "chosen_policy_test_review_share": p["chosen_policy"]["test"]["review_share"],
            "chosen_policy_test_shifts_per_day": p["chosen_policy"]["test"]["shifts_per_day"],
        })
    return rows


def comparison(val: BagScores, test: BagScores, costs: Costs = DEFAULT_COSTS) -> list[dict]:
    """Cost per 1,000 bags (val and test) and daily staffing (test shares) of the alternatives.

    "Do nothing", "a person checks every image" and "search every bag" send every bag one way
    (thresholds of +/- inf).
    """
    k, inf = costs, np.inf
    p = policies(val, test, k)
    a, alt, ch = (p[n]["test"] for n in ("model_alone", "alternative_0_7", "chosen_policy"))
    rows = [
        ("Do nothing (every bag passes)", (inf, inf)),
        ("A screener checks every image", (-inf, inf)),
        ("Search every bag by hand", (-inf, -inf)),
        (f"Model alone, one threshold (t = {a['t_low']:g})", (a["t_low"], a["t_high"])),
        (f"Earlier proposal: model + screener ({alt['t_low']:g} / {alt['t_high']:g})", (alt["t_low"], alt["t_high"])),
        (f"Chosen policy: model + screener ({ch['t_low']:g} / {ch['t_high']:g})", (ch["t_low"], ch["t_high"])),
    ]
    return [{"option": name, "val_cost_per_1000": evaluate(val, *pair, k)["cost_per_1000"],
             **_row(evaluate(test, *pair, k))} for name, pair in rows]


def _row(result: dict) -> dict:
    keys = ("cost_per_1000", "review_share", "search_share", "reviews_per_day", "shifts_per_day",
            "searches_per_day", "search_shifts_per_day")
    return {key: result[key] for key in keys}


# -------------------------------------------------------------------------- plot
def eur_axis(ax) -> None:
    """Log y axis with euro tick labels (EUR 200, EUR 1K, EUR 50K, EUR 1M)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

    def fmt(v, _):
        return f"€{v / 1e6:g}M" if v >= 1e6 else f"€{v / 1e3:g}K" if v >= 1e3 else f"€{v:g}"

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))


def plot_cost_curve(val: BagScores, test: BagScores, path: Path | None = None, costs: Costs = DEFAULT_COSTS):
    """Policy cost per 1,000 bags as t_low varies (t_high fixed at the chosen value), on val and test,
    with the model-alone (one threshold) curve on test for reference. Chosen thresholds are marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    ink, ink2, muted, gridc, surface = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
    blue, orange, grey = "#2a78d6", "#d6532a", "#a3a29b"
    p = policies(val, test, costs)
    lo, hi = p["chosen_policy"]["test"]["t_low"], p["chosen_policy"]["test"]["t_high"]
    t_alone = p["model_alone"]["test"]["t_low"]

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.step(THRESHOLDS, sweep(test, costs), where="post", color=grey, lw=1.5, label="Model alone (one threshold), test")
    ax.step(THRESHOLDS, policy_curve(val, hi, costs), where="post", color=orange, lw=1.8,
            label=f"Model + screener, t_high = {hi:g}, validation (chooses)")
    ax.step(THRESHOLDS, policy_curve(test, hi, costs), where="post", color=blue, lw=2.2,
            label=f"Model + screener, t_high = {hi:g}, test (reported)")
    for split, scores, colour in (("val", val, orange), ("test", test, blue)):
        cost = p["chosen_policy"][split]["cost_per_1000"]
        ax.plot([lo], [cost], "o", ms=9, color=colour, mec=surface, mew=2, zorder=3)
    c_test = p["chosen_policy"]["test"]["cost_per_1000"]
    ax.annotate(f"Chosen t_low = {lo:g}: €{c_test:,.0f} per 1,000 bags on test", (lo, c_test),
                xytext=(-8, -30), textcoords="offset points", ha="right", va="top", fontsize=9.5, color=ink,
                arrowprops=dict(arrowstyle="-", color=ink2, lw=0.8))
    c_alone = p["model_alone"]["test"]["cost_per_1000"]
    ax.plot([t_alone], [c_alone], "s", ms=7, color=grey, mec=ink2, mew=1, zorder=4)
    ax.annotate(f"Model alone\nt = {t_alone:g}: €{c_alone:,.0f}", (t_alone, c_alone),
                xytext=(10, -30), textcoords="offset points", ha="left", va="top", fontsize=9, color=ink2,
                arrowprops=dict(arrowstyle="-", color=ink2, lw=0.8))
    ax.axvline(hi, color=ink2, ls="--", lw=1)
    ax.text(hi, 0.97, f" t_high = {hi:g}", transform=ax.get_xaxis_transform(), fontsize=9, color=ink2, va="top")

    ax.set_xscale("log")
    ax.set_xlim(THRESHOLDS[0] * 0.9, 1.0)
    ax.xaxis.set_major_locator(FixedLocator([0.001, 0.01, 0.1, 0.5, 1.0]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    eur_axis(ax)
    ax.set_ylim(40, 2.5e5)
    ax.set_xlabel("t_low, log scale (one-threshold curve: t)", color=ink2, fontsize=9.5, labelpad=8)
    ax.set_ylabel("Expected cost per 1,000 bags (log scale)", color=ink2, fontsize=9.5, labelpad=8)
    ax.grid(True, which="major", color=gridc, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=muted, labelsize=9, length=0, pad=6)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.text(0.012, 0.97, f"Chosen on validation: pass below {lo:g}, screener from {lo:g} to {hi:g}, search from {hi:g}",
             ha="left", va="top", fontsize=12.5, fontweight="bold", color=ink)
    fig.text(0.012, 0.915,
             f"Base rate 1 in {1 / costs.base_rate:,.0f}; threat mix "
             + ", ".join(f"{n.lower()} {100 * v:g}%" for n, v in THREAT_MIX.items())
             + f". Search €{costs.fp:g}, screener check €{costs.review:.2f}, catches {100 * costs.reviewer_catch:g}%."
             "\nClean bags are approximated by the boxes drawn on empty content of the images.",
             ha="left", va="top", fontsize=9.5, color=ink2, linespacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    if path is not None:
        fig.savefig(path, facecolor=surface)
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- run
def run(predictions: Path = PREDICTIONS, out_dir: Path = ARTIFACTS_DIR, val_predictions: Path = VAL_PREDICTIONS) -> dict:
    val, test = load_bag_scores(val_predictions), load_bag_scores(predictions)
    k = DEFAULT_COSTS
    p = policies(val, test, k)
    chosen = p["chosen_policy"]
    pair = (chosen["test"]["t_low"], chosen["test"]["t_high"])
    t_alone = p["model_alone"]["test"]["t_low"]

    def split_info(s: BagScores) -> dict:
        return {
            "threat": len(s.threat),
            "threat_by_most_severe_class": {n: int((s.threat_class == n).sum()) for n in SEVERITY},
            "weight_per_bag": {n: float(s.threat_weight[s.threat_class == n][0]) for n in SEVERITY
                               if (s.threat_class == n).any()},
            "threat_no_box": int((s.threat < THRESHOLDS[0]).sum()),
            "clean_proxy": len(s.clean),
        }

    result = {
        "method": {
            "unit": "one bag = one image",
            "selection": "thresholds chosen on the validation split (2-decimal grid 0.01-0.99), costs reported on the test split",
            "threat_score": "highest confidence among all boxes in the image, any class, no IoU matching (0 if none)",
            "miss_cost": "a threat bag's miss cost is that of its most severe ground-truth class (gun > knife > scissors > pliers / wrench)",
            "threat_weighting": "each threat bag weighs threat_mix[class] / (threat bags of that class in the split); all means over threat bags use these weights",
            "clean_score": f"stand-in: highest confidence among boxes with IoU <= {CLEAN_MAX_IOU} with every ground-truth box",
            "model_alone": "searched if score >= t, else pass",
            "design": "pass if score < t_low, review if t_low <= score < t_high, search if score >= t_high",
            "chosen_policy": "design with the cost-optimal (t_low, t_high) pair on validation",
            "alternative_0_7": f"design with t_high = {T_HIGH} and the cost-optimal t_low for it on validation (earlier proposal)",
            "reviewer": "catches a reviewed threat with probability reviewer_catch; never sends a clean bag to search (reviewer_sends_clean = 0)",
            "ties": "lowest t_low, then lowest t_high, among equal costs",
            "counts": "plain bag counts in the split (not weighted)",
        },
        "assumptions": {
            **asdict(k),
            "threat_mix": THREAT_MIX,
            "review_seconds": REVIEW_SECONDS,
            "screener_eur_per_hour": SCREENER_EUR_PER_HOUR,
            "review_share_of_time": REVIEW_SHARE_OF_TIME,
            "shift_hours": SHIFT_HOURS,
            "bags_per_day": BAGS_PER_DAY,
            "search_minutes": SEARCH_MINUTES,
            "search_officer_time_only": SEARCH_OFFICER_ONLY,
            "clean_max_iou": CLEAN_MAX_IOU,
        },
        "bags": {"val": split_info(val), "test": split_info(test)},
        "chosen_policy": chosen["test"],
        "chosen_policy_val": chosen["val"],
        "alternative_0_7": p["alternative_0_7"]["test"],
        "alternative_0_7_val": p["alternative_0_7"]["val"],
        "model_alone": p["model_alone"]["test"],
        "model_alone_val": p["model_alone"]["val"],
        "plateau_val": {"exact_ties": plateau(val, k, 0.0), "within_1pct": plateau(val, k, 0.01),
                        "within_5pct": plateau(val, k, 0.05)},
        "comparison": comparison(val, test, k),
        "break_even": {
            # Screener catch rate below which the chosen pair costs more than the model alone (each at its val-chosen thresholds)
            "reviewer_catch_vs_model_alone_test": reviewer_break_even(test, pair, t_alone, k),
            "reviewer_catch_vs_model_alone_val": reviewer_break_even(val, pair, t_alone, k),
            # Values at which the val-chosen t_low first moves (None: not within the range searched)
            "t_low_moves_miss_scale_down": t_low_moves(val, "miss_scale", 1.0, 1e-4, k),
            "t_low_moves_miss_scale_up": t_low_moves(val, "miss_scale", 1.0, 1e3, k),
            "t_low_moves_base_rate_down": t_low_moves(val, "base_rate", k.base_rate, 1e-7, k),
            "t_low_moves_base_rate_up": t_low_moves(val, "base_rate", k.base_rate, 0.5, k),
        },
        # Same thresholds as chosen, but the screener sends 10% of the clean bags they check on to a search
        "screener_sends_10pct_clean_at_chosen": {
            split: evaluate(s, *pair, replace(k, reviewer_sends_clean=0.1))["cost_per_1000"]
            for split, s in (("val", val), ("test", test))
        },
        "sensitivity": sensitivity(val, test, k),
        "bag_scores": {
            "val": {"threat": val.threat.round(4).tolist(), "clean": val.clean.round(4).tolist()},
            "test": {"threat": test.threat.round(4).tolist(), "clean": test.clean.round(4).tolist()},
        },
        "curve_test": {
            "threshold": THRESHOLDS.tolist(),
            "model_alone_cost_per_1000": sweep(test, k).round(2).tolist(),
            f"policy_t_high_{pair[1]:g}_cost_per_1000": [None if np.isnan(v) else round(float(v), 2)
                                                        for v in policy_curve(test, pair[1], k)],
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cost_sweep.json").write_text(json.dumps(result, indent=2))
    plot_cost_curve(val, test, out_dir / "cost_curve.png", k)
    return result


def _print_summary(r: dict) -> None:
    def line(name, e, v):
        print(f"{name:<30} t_low {e['t_low']:<5g} t_high {e['t_high']:<5g} val EUR {v['cost_per_1000']:>9,.2f}  "
              f"test EUR {e['cost_per_1000']:>9,.2f} per 1,000  review {100 * e['review_share']:5.2f}%  "
              f"{e['reviews_per_day']:>7,.0f}/day {e['shifts_per_day']:5.1f} shifts")

    line("Chosen policy", r["chosen_policy"], r["chosen_policy_val"])
    line("Alternative (t_high = 0.7)", r["alternative_0_7"], r["alternative_0_7_val"])
    line("Model alone", r["model_alone"], r["model_alone_val"])
    print("\nPlateau on val:")
    for name, pl in r["plateau_val"].items():
        print(f"  {name:<12} {pl['pairs']:>4} pairs  t_low {pl['t_low_range']}  t_high {pl['t_high_range']}  "
              f"t_low at chosen t_high {pl['t_low_range_at_chosen_t_high']}  "
              f"t_high at chosen t_low {pl['t_high_range_at_chosen_t_low']}")
    print("\nChosen policy, test counts:", r["chosen_policy"]["counts"])
    print()
    for row in r["comparison"]:
        print(f"{row['option']:<50} val EUR {row['val_cost_per_1000']:>11,.2f}  test EUR {row['cost_per_1000']:>11,.2f}  "
              f"review {100 * row['review_share']:6.2f}%  search {100 * row['search_share']:6.2f}%  "
              f"{row['reviews_per_day']:>7,.0f} reviews/day {row['shifts_per_day']:6.1f} shifts  "
              f"{row['searches_per_day']:>7,.0f} searches/day {row['search_shifts_per_day']:6.1f} shifts")
    print("\nBreak-even:")
    for key, v in r["break_even"].items():
        print(f"  {key}: {v}")
    print("\nSensitivity (thresholds re-chosen on val, cost on test):")
    for s in r["sensitivity"]:
        print(f"  {s['case']:<52} alone t={s['model_alone_t_low']:<5g} EUR {s['model_alone_test_cost_per_1000']:>9,.0f} | "
              f"policy {s['chosen_policy_t_low']:g}/{s['chosen_policy_t_high']:g} "
              f"val EUR {s['chosen_policy_val_cost_per_1000']:>9,.0f} test EUR {s['chosen_policy_test_cost_per_1000']:>9,.0f} "
              f"review {100 * s['chosen_policy_test_review_share']:.1f}% | 0.7: t_low {s['alternative_0_7_t_low']:g} "
              f"EUR {s['alternative_0_7_test_cost_per_1000']:>9,.0f}")


if __name__ == "__main__":
    _print_summary(run())
    print(f"\nWrote {ARTIFACTS_DIR / 'cost_sweep.json'} and {ARTIFACTS_DIR / 'cost_curve.png'}")
