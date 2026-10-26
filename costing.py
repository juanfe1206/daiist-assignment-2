"""
Assignment 2 cost sweep: what each threshold costs per 1,000 bags.

    uv run python costing.py     # reads artifacts/test_predictions.json
                                 # -> artifacts/cost_sweep.json, artifacts/cost_curve.png

Also run at the end of train.py, and imported by the dashboard (app.py).

Method (REPORT.md, "Cost matrix and operating point" and "What a person still
has to check"):

- The decision unit is one bag, i.e. one test image.
- Threat-bag score: the highest confidence among ALL boxes in the image (any
  class, wherever they are). The decision is whether to open the bag, so if the
  model fires anything on a bag that holds a threat, the bag is searched and the
  threat found. A threat is missed only when no box reaches the threshold.
- The test split has no clean bags, so a clean bag is approximated by the empty
  content of each test image: the highest confidence among the boxes that do
  not overlap any real object (IoU <= CLEAN_MAX_IOU with every ground-truth
  box). Those 156 values stand in for 156 clean bags. It is only an
  approximation: the empty part of a threat bag is smaller and more cluttered
  than a real clean bag.
- Model alone: a bag is searched if its score >= t, otherwise it passes.
- Model + person: score < t_low passes, t_low <= score < t_high goes to a
  screener for a second look, score >= t_high goes straight to a manual search.
  The screener catches a reviewed threat with probability REVIEWER_CATCH (then
  it is searched) and clears a reviewed clean bag (no false alarms by the
  reviewer, to keep it simple).
- Expected cost per 1,000 bags =
      1000 x [ p x (threat bag cost) + (1 - p) x (clean bag cost) ]
  where p is the base rate and each bag cost is the share of bags in each
  outcome times the euro cost of that outcome.
"""

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
PREDICTIONS = ARTIFACTS_DIR / "test_predictions.json"
MODEL = "finetuned_yolo11n"

# ------------------------------------------------------------- costs (REPORT.md)
# "Cost matrix and operating point", cost table
C_FP = 2.0  # clean bag searched for nothing (EUR)
C_FN = 1_000_000.0  # threat passes (EUR, a floor: the Munich 2018 breach)
C_TP = 2.0  # threat found: the same manual search (EUR)
C_TN = 0.0  # clean bag passes (EUR)
# "Cost matrix and operating point", base rate: 1 bag in 1,000 holds a threat
BASE_RATE = 0.001

# "What a person still has to check", review load
REVIEW_SECONDS = 30  # a second look at one image (assumption in REPORT.md)
SCREENER_EUR_PER_HOUR = 20.0  # loaded screener cost ("Cost matrix" section)
REVIEW_SHARE_OF_TIME = 20 / 30  # 20 min of image review, then a 10 min break
SHIFT_HOURS = 8  # -> 8 x 20/30 = 5.3 hours of image review per shift
BAGS_PER_DAY = 90_700  # Madrid-Barajas departing bags a day
# Paid cost of one review, breaks included: 30 s x EUR 20/h / (20/30) = EUR 0.25.
# Same basis as REPORT.md's "person checks everything" row (EUR 250 per 1,000).
REVIEW_COST = REVIEW_SECONDS / 3600 * SCREENER_EUR_PER_HOUR / REVIEW_SHARE_OF_TIME

# ASSUMPTION (not in REPORT.md): a screener catches 90% of the threats in the
# images sent to review. The sensitivity table shows 80%, 99% and 100%.
REVIEWER_CATCH = 0.9

# ------------------------------------------------------------------ sweep set-up
CLEAN_MAX_IOU = 0.1  # a box counts as "on empty content" if its IoU with every real object is <= this
THRESHOLDS = np.round(np.arange(1, 991) / 1000, 3)  # 0.001 to 0.990
PROVISIONAL_T_HIGH = 0.7  # the student's provisional upper threshold


@dataclass(frozen=True)
class Costs:
    """Everything the expected cost depends on. Change one field with dataclasses.replace."""

    fp: float = C_FP
    fn: float = C_FN
    tp: float = C_TP
    tn: float = C_TN
    base_rate: float = BASE_RATE
    review: float = REVIEW_COST
    reviewer_catch: float = REVIEWER_CATCH


DEFAULT_COSTS = Costs()


@dataclass(frozen=True)
class BagScores:
    """One score per test bag: threat = boxes on real objects, clean = boxes on background."""

    threat: np.ndarray
    clean: np.ndarray


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


def load_bag_scores(path: Path = PREDICTIONS, model: str = MODEL, clean_max_iou: float = CLEAN_MAX_IOU) -> BagScores:
    images = json.loads(Path(path).read_text())["images"]
    pairs = [
        image_scores(np.array(im["gt"]).reshape(-1, 5), np.array(im[model]).reshape(-1, 6), clean_max_iou)
        for im in images
    ]
    threat, clean = np.array(pairs).T
    return BagScores(threat=threat, clean=clean)


# --------------------------------------------------------------------- outcomes
def share_at_or_above(scores: np.ndarray, t) -> np.ndarray:
    """Share of scores >= t, for a scalar or an array of thresholds."""
    ordered = np.sort(scores)
    return 1 - np.searchsorted(ordered, np.asarray(t, dtype=float), side="left") / len(ordered)


def outcomes(scores: BagScores, t_low, t_high=None) -> dict:
    """Share of threat bags and of clean bags that pass, go to review, or go to search.

    t_high=None means the model alone (t_high = t_low, no review band).
    Works on scalars or broadcastable arrays of thresholds.
    """
    t_high = t_low if t_high is None else t_high
    out = {}
    for kind in ("threat", "clean"):
        s = getattr(scores, kind)
        above_low, above_high = share_at_or_above(s, t_low), share_at_or_above(s, t_high)
        out[kind] = {"pass": 1 - above_low, "review": above_low - above_high, "search": above_high}
    return out


def cost_per_1000(o: dict, costs: Costs = DEFAULT_COSTS):
    """Expected EUR per 1,000 bags for outcome shares from outcomes()."""
    t, c, k = o["threat"], o["clean"], costs
    reviewed_threat = k.review + k.reviewer_catch * k.tp + (1 - k.reviewer_catch) * k.fn
    threat_bag = t["pass"] * k.fn + t["review"] * reviewed_threat + t["search"] * k.tp
    clean_bag = c["pass"] * k.tn + c["review"] * k.review + c["search"] * k.fp
    return 1000 * (k.base_rate * threat_bag + (1 - k.base_rate) * clean_bag)


def review_share(o: dict, base_rate: float = BASE_RATE):
    """Share of all bags sent to a person."""
    return base_rate * o["threat"]["review"] + (1 - base_rate) * o["clean"]["review"]


def staffing(share: float, bags_per_day: int = BAGS_PER_DAY) -> dict:
    """Reviews a day, image-review hours a day and 8-hour screener shifts (20-minute rule)."""
    reviews = share * bags_per_day
    hours = reviews * REVIEW_SECONDS / 3600
    return {
        "reviews_per_day": reviews,
        "review_hours_per_day": hours,
        "shifts_per_day": hours / (SHIFT_HOURS * REVIEW_SHARE_OF_TIME),
    }


def evaluate(scores: BagScores, t_low: float, t_high: float | None = None, costs: Costs = DEFAULT_COSTS) -> dict:
    """Everything the dashboard shows for one pair of thresholds (t_high=None: model alone)."""
    o = outcomes(scores, t_low, t_high)
    p = costs.base_rate
    t, c = o["threat"], o["clean"]
    share = float(review_share(o, p))
    return {
        "t_low": float(t_low),
        "t_high": float(t_low if t_high is None else t_high),
        "cost_per_1000": float(cost_per_1000(o, costs)),
        "review_share": share,
        "search_share": float(p * t["search"] + (1 - p) * c["search"]),
        # Expected bags per 1,000 in each cell (rows: truth, columns: what happens)
        "per_1000_bags": {
            "threat": {k: float(1000 * p * v) for k, v in t.items()},
            "clean": {k: float(1000 * (1 - p) * v) for k, v in c.items()},
        },
        # Test-split counts behind those shares (156 threat bags, 156 clean proxies)
        "test_counts": {
            kind: {k: int(round(float(v) * len(getattr(scores, kind)))) for k, v in o[kind].items()}
            for kind in ("threat", "clean")
        },
        **staffing(share),
    }


# ------------------------------------------------------------------------ sweeps
def sweep(scores: BagScores, costs: Costs = DEFAULT_COSTS, thresholds: np.ndarray = THRESHOLDS) -> np.ndarray:
    """Model-alone cost per 1,000 bags at every threshold."""
    return cost_per_1000(outcomes(scores, thresholds), costs)


def best_threshold(scores: BagScores, costs: Costs = DEFAULT_COSTS, thresholds: np.ndarray = THRESHOLDS) -> float:
    """Model-alone threshold with the lowest cost. On a tie, the lowest threshold (the safer side)."""
    return float(thresholds[np.argmin(sweep(scores, costs, thresholds))])


def best_pair(
    scores: BagScores,
    costs: Costs = DEFAULT_COSTS,
    thresholds: np.ndarray = THRESHOLDS,
    t_high: float | None = None,
) -> tuple[float, float]:
    """(t_low, t_high) with the lowest cost, t_low <= t_high. Pass t_high to fix it and search t_low only.

    On a tie, the lowest t_low, then the lowest t_high.
    """
    highs = thresholds if t_high is None else np.array([t_high])
    lo, hi = thresholds[:, None], highs[None, :]
    grid = np.where(lo <= hi, cost_per_1000(outcomes(scores, lo, hi), costs), np.inf)
    i, j = np.unravel_index(np.argmin(grid), grid.shape)
    return float(thresholds[i]), float(highs[j])


def break_even(scores: BagScores, field: str, low: float, high: float, costs: Costs = DEFAULT_COSTS) -> float:
    """Value of one Costs field (fn or base_rate) below which the best model-alone threshold rises.

    Bisection on a log scale between low and high (the best threshold only rises as either one falls).
    """
    base = best_threshold(scores, costs)
    for _ in range(60):
        mid = float(np.sqrt(low * high))
        if best_threshold(scores, replace(costs, **{field: mid})) == base:
            high = mid
        else:
            low = mid
    return high


def reviewer_break_even(scores: BagScores, t_low: float, t_high: float, costs: Costs = DEFAULT_COSTS) -> float:
    """Reviewer catch rate at which (t_low, t_high) costs the same as the best model-alone threshold.

    The design's cost is a straight line in the catch rate, so two points give it.
    """
    target = evaluate(scores, best_threshold(scores, costs), None, costs)["cost_per_1000"]
    c0, c1 = (evaluate(scores, t_low, t_high, replace(costs, reviewer_catch=r))["cost_per_1000"] for r in (0.0, 1.0))
    return float((c0 - target) / (c0 - c1)) if c0 != c1 else float("nan")


def sensitivity(scores: BagScores, costs: Costs = DEFAULT_COSTS, predictions: Path = PREDICTIONS) -> list[dict]:
    """Best thresholds and costs when one assumption changes (one row per change)."""
    cases = [("baseline", {})]
    cases += [(f"missed threat EUR {v:,.0f}", {"fn": v}) for v in (10_000, 100_000, 10_000_000)]
    cases += [(f"base rate {100 * v:g}%", {"base_rate": v}) for v in (0.0001, 0.01)]
    cases += [(f"reviewer catches {100 * v:g}%", {"reviewer_catch": v}) for v in (0.8, 0.99, 1.0)]
    # The clean stand-in rule changes the scores themselves, not the costs
    cases += [(f"clean stand-in IoU <= {v:g}", {"clean_max_iou": v}) for v in (0.0, 0.1, 0.3)]
    rows = []
    for label, change in cases:
        if "clean_max_iou" in change:
            scores_now, k = load_bag_scores(predictions, clean_max_iou=change["clean_max_iou"]), costs
        else:
            scores_now, k = scores, replace(costs, **change)
        rows.append(_sensitivity_row(label, change, scores_now, k))
    return rows


def _sensitivity_row(label: str, change: dict, scores: BagScores, k: Costs) -> dict:
    t = best_threshold(scores, k)
    t_low, t_high = best_pair(scores, k)
    t_low_07, _ = best_pair(scores, k, t_high=PROVISIONAL_T_HIGH)
    design, at_07 = evaluate(scores, t_low, t_high, k), evaluate(scores, t_low_07, PROVISIONAL_T_HIGH, k)
    return {
        "case": label,
        "change": change,
        "model_alone_t": t,
        "model_alone_cost_per_1000": evaluate(scores, t, None, k)["cost_per_1000"],
        "design_t_low": t_low,
        "design_t_high": t_high,
        "design_cost_per_1000": design["cost_per_1000"],
        "design_review_share": design["review_share"],
        "t_high_0.7_t_low": t_low_07,
        "t_high_0.7_cost_per_1000": at_07["cost_per_1000"],
        "t_high_0.7_review_share": at_07["review_share"],
    }


def comparison(scores: BagScores, costs: Costs = DEFAULT_COSTS) -> list[dict]:
    """Cost per 1,000 bags of the alternatives in REPORT.md's comparison table, plus searching every bag."""
    k = costs

    def everyone(outcome: str) -> dict:
        return {kind: {o: float(o == outcome) for o in ("pass", "review", "search")} for kind in ("threat", "clean")}

    t = best_threshold(scores, k)
    t_low, t_high = best_pair(scores, k)
    t_low_07, _ = best_pair(scores, k, t_high=PROVISIONAL_T_HIGH)
    return [
        {"option": "Do nothing", "cost_per_1000": float(cost_per_1000(everyone("pass"), k)), "review_share": 0.0},
        {"option": "A person checks everything", "cost_per_1000": float(cost_per_1000(everyone("review"), k)), "review_share": 1.0},
        {"option": "Search every bag by hand", "cost_per_1000": float(cost_per_1000(everyone("search"), k)), "review_share": 0.0},
        {"option": f"Model alone (t = {t:g})", **_cost_and_share(evaluate(scores, t, None, k))},
        {"option": f"Model + review, cost-optimal ({t_low:g} / {t_high:g})", **_cost_and_share(evaluate(scores, t_low, t_high, k))},
        {"option": f"Model + review, t_high = 0.7 ({t_low_07:g} / 0.7)", **_cost_and_share(evaluate(scores, t_low_07, PROVISIONAL_T_HIGH, k))},
    ]


def _cost_and_share(result: dict) -> dict:
    return {"cost_per_1000": result["cost_per_1000"], "review_share": result["review_share"]}


# -------------------------------------------------------------------------- plot
def plot_cost_curve(scores: BagScores, path: Path | None = None, costs: Costs = DEFAULT_COSTS, mark: dict | None = None):
    """Model-alone cost per 1,000 bags against the threshold (log-log), with named thresholds marked.

    mark: {label: threshold}, default the lowest-cost threshold. Saves to path if given and
    returns the matplotlib figure, so the dashboard can show it directly.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    ink, ink2, muted, grid, axis, surface = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    series = "#2a78d6"
    curve = sweep(scores, costs)
    mark = {"Lowest cost": best_threshold(scores, costs)} if mark is None else mark

    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.step(THRESHOLDS, curve, where="post", color=series, linewidth=2, solid_joinstyle="round", solid_capstyle="round")
    ax.set_xscale("log")
    ax.set_yscale("log")

    for i, (label, t) in enumerate(mark.items()):
        cost = float(cost_per_1000(outcomes(scores, t), costs))
        ax.plot([t], [cost], "o", markersize=9, color=series, markeredgecolor=surface, markeredgewidth=2, zorder=3)
        above = i > 0  # the first mark is labelled below-left of its point, the others above-left
        ax.annotate(
            f"{label}\nt = {t:g}: EUR {cost:,.0f} per 1,000 bags",
            (t, cost), xytext=(-12, 12) if above else (-12, -12), textcoords="offset points",
            ha="right", va="bottom" if above else "top", fontsize=9.5, color=ink,
        )

    ax.set_xlim(THRESHOLDS[0] * 0.9, 1.0)
    ax.xaxis.set_major_locator(FixedLocator([0.001, 0.01, 0.1, 0.5, 1.0]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_locator(FixedLocator([100, 200, 500, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "EUR 1M" if v >= 1e6 else f"EUR {v / 1000:,.0f}K" if v >= 1e3 else f"EUR {v:,.0f}"))
    ax.set_ylim(min(curve.min() * 0.3, 1e6), 1.6e6)
    ax.set_xlabel("Confidence threshold (log scale). A bag is searched if its highest box is at or above it.",
                  color=ink2, fontsize=9.5, labelpad=8)
    ax.set_ylabel("Expected cost per 1,000 bags (log scale)", color=ink2, fontsize=9.5, labelpad=8)
    ax.grid(True, which="major", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(axis)
    ax.tick_params(colors=muted, labelsize=9, length=0, pad=6)

    best = best_threshold(scores, costs)
    title = f"Cost is lowest at {best:g}"
    if share_at_or_above(scores.threat, best) == 1:
        title += ", the highest threshold that still flags every test threat bag"
    fig.text(0.012, 0.965, title,
             ha="left", va="top", fontsize=12.5, fontweight="bold", color=ink)
    fig.text(
        0.012, 0.915,
        f"Model alone, 156 test bags. Base rate {100 * costs.base_rate:g}%, EUR {costs.fn:,.0f} per missed threat, "
        f"EUR {costs.fp:g} per search.\nClean bags are approximated by the boxes drawn on empty content of the test images.",
        ha="left", va="top", fontsize=9.5, color=ink2, linespacing=1.4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    if path is not None:
        fig.savefig(path, facecolor=surface)
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- run
def run(predictions: Path = PREDICTIONS, out_dir: Path = ARTIFACTS_DIR) -> dict:
    scores = load_bag_scores(predictions)
    k = DEFAULT_COSTS
    t = best_threshold(scores, k)
    t_low, t_high = best_pair(scores, k)
    t_low_07, _ = best_pair(scores, k, t_high=PROVISIONAL_T_HIGH)
    curve = sweep(scores, k)

    result = {
        "method": {
            "unit": "one bag = one test image",
            "threat_score": "highest confidence among all boxes in the image, any class, no IoU matching (0 if none)",
            "clean_score": f"stand-in: highest confidence among boxes with IoU <= {CLEAN_MAX_IOU} with every ground-truth box",
            "model_alone": "searched if score >= t, else pass",
            "design": "pass if score < t_low, review if t_low <= score < t_high, search if score >= t_high",
            "reviewer": "catches a reviewed threat with probability reviewer_catch, clears a reviewed clean bag",
            "ties": "lowest threshold among equal costs",
        },
        "assumptions": {
            **asdict(k),
            "review_seconds": REVIEW_SECONDS,
            "screener_eur_per_hour": SCREENER_EUR_PER_HOUR,
            "review_share_of_time": REVIEW_SHARE_OF_TIME,
            "shift_hours": SHIFT_HOURS,
            "bags_per_day": BAGS_PER_DAY,
            "clean_max_iou": CLEAN_MAX_IOU,
        },
        "test_bags": {
            "threat": len(scores.threat),
            "threat_no_box": int((scores.threat < THRESHOLDS[0]).sum()),
            "clean_proxy": len(scores.clean),
        },
        "bag_scores": {"threat": scores.threat.round(4).tolist(), "clean": scores.clean.round(4).tolist()},
        "curve": {"threshold": THRESHOLDS.tolist(), "cost_per_1000": curve.round(2).tolist()},
        "model_alone": evaluate(scores, t, None, k),
        "model_alone_at_0.7": evaluate(scores, PROVISIONAL_T_HIGH, None, k),
        "design_optimal": evaluate(scores, t_low, t_high, k),
        "design_t_high_0.7": evaluate(scores, t_low_07, PROVISIONAL_T_HIGH, k),
        "comparison": comparison(scores, k),
        "break_even": {
            "fn_eur": break_even(scores, "fn", 1.0, k.fn, k),
            "base_rate": break_even(scores, "base_rate", 1e-9, k.base_rate, k),
            "reviewer_catch_for_t_high_0.7": reviewer_break_even(scores, t_low_07, PROVISIONAL_T_HIGH, k),
        },
        "sensitivity": sensitivity(scores, k, predictions),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cost_sweep.json").write_text(json.dumps(result, indent=2))
    plot_cost_curve(scores, out_dir / "cost_curve.png", k, {"Lowest cost": t, "0.7 used alone (no review)": PROVISIONAL_T_HIGH})
    return result


def _print_summary(r: dict) -> None:
    def line(name, e):
        print(f"{name:<34} t_low {e['t_low']:<6g} t_high {e['t_high']:<6g} EUR {e['cost_per_1000']:>11,.2f} per 1,000  "
              f"review {100 * e['review_share']:6.2f}%  {e['reviews_per_day']:>8,.0f}/day  "
              f"{e['shifts_per_day']:6.1f} shifts")

    line("Model alone (best t)", r["model_alone"])
    line("Model alone at 0.7", r["model_alone_at_0.7"])
    line("Model + review (cost-optimal)", r["design_optimal"])
    line("Model + review (t_high = 0.7)", r["design_t_high_0.7"])
    print()
    for row in r["comparison"]:
        print(f"{row['option']:<50} EUR {row['cost_per_1000']:>13,.2f}  review {100 * row['review_share']:.2f}%")
    print(f"\nBreak-even: missed threat EUR {r['break_even']['fn_eur']:,.0f}, base rate {r['break_even']['base_rate']:.3g}, "
          f"reviewer catch for 0.7 {r['break_even']['reviewer_catch_for_t_high_0.7']:.4f}")
    print("\nSensitivity:")
    for s in r["sensitivity"]:
        print(f"  {s['case']:<28} alone t={s['model_alone_t']:<6g} EUR {s['model_alone_cost_per_1000']:>11,.0f} | "
              f"design {s['design_t_low']:g}/{s['design_t_high']:g} EUR {s['design_cost_per_1000']:>11,.0f} "
              f"review {100 * s['design_review_share']:.1f}% | 0.7: t_low {s['t_high_0.7_t_low']:g} "
              f"EUR {s['t_high_0.7_cost_per_1000']:>11,.0f} review {100 * s['t_high_0.7_review_share']:.1f}%")


if __name__ == "__main__":
    _print_summary(run())
    print(f"\nWrote {ARTIFACTS_DIR / 'cost_sweep.json'} and {ARTIFACTS_DIR / 'cost_curve.png'}")
