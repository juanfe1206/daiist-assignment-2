"""
Assignment 2: Gradio dashboard for X-ray baggage screening.

    uv run python main.py app      # then open http://127.0.0.1:7860

Built only from what the training stage already produced. Nothing is trained here:

- weights/best.pt                    finetuned yolo11n (Gun, Knife, Pliers, Scissors, Wrench)
- artifacts/test_predictions.json    test-split boxes, turned into bag scores by costing.py
- artifacts/metrics.json             per-class AP50, AP50-95, P, R for the model and both baselines
- artifacts/ultralytics_test/        Ultralytics' own test plots

Policy (REPORT.md): a bag whose highest box scores below t_low passes, from t_low
up to 0.7 a screener checks the X-ray image and decides, and from 0.7 up it goes
straight to a manual search. t_low is the cost-optimal lower threshold for
t_high = 0.7 (costing.chosen_pair). The alternative compared is one threshold
(the best single threshold, no review band).

Layout: two threshold sliders at the top drive every view below them.
  1. Cost and review load   bag-level confusion matrix per 1,000 bags, cost, review share, cost curve
  2. Screen a bag           upload an X-ray, see boxes and the bag decision
  3. Per-class results      finetuned model against the two baselines
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; figures go straight to Gradio
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from ultralytics import YOLO

import costing

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "best.pt"
ARTIFACTS = ROOT / "artifacts"
PORT = int(os.environ.get("APP_PORT", "7860"))  # check.py reads the same variable

IMGSZ = 416  # same as train.py: the dataset is 416x416
PRED_CONF = 0.001  # same as train.py: keep almost every box, the thresholds decide later


DECISIONS = {"pass": "Pass", "review": "Human review", "search": "Manual search"}

# ------------------------------------------------------------- load once at startup
MODEL = YOLO(str(WEIGHTS))
MODEL.predict(np.zeros((IMGSZ, IMGSZ, 3), np.uint8), imgsz=IMGSZ, verbose=False)  # warm-up: first upload is fast
SCORES = costing.load_bag_scores()  # 156 threat bags and 156 clean stand-ins from the test split
CURVE = costing.sweep(SCORES)  # model-alone cost per 1,000 bags at every threshold
METRICS = json.loads((ARTIFACTS / "metrics.json").read_text())
K = costing.DEFAULT_COSTS

CHOSEN_T_LOW, CHOSEN_T_HIGH = costing.chosen_pair(SCORES, K)  # the policy in REPORT.md: 0.15 / 0.7
ALTERNATIVE_T = costing.best_threshold(SCORES, K)  # the alternative compared: one threshold at 0.15
CHOSEN = f"{CHOSEN_T_LOW:g} / {CHOSEN_T_HIGH:g}"

ASSUMPTIONS = (
    f"**Assumptions** (REPORT.md): base rate **1 threat bag in {1 / K.base_rate:,.0f}**. "
    f"Missed threat, by the most severe object in the bag: "
    + ", ".join(f"{name.lower()} **€{K.miss_cost(name):,.0f}**" for name in costing.SEVERITY)
    + f". Clean bag searched **€{K.fp:g}**, threat found **€{K.tp:g}**, "
    f"clean bag passed **€{K.tn:g}**, one human review **€{K.review:.2f}** "
    f"(30 s at €20/h with breaks), reviewer catches **{100 * K.reviewer_catch:g}%** of reviewed threats. "
    f"Staffing assumes {costing.BAGS_PER_DAY:,} bags a day (Madrid-Barajas) and 8-hour shifts.\n\n"
    f"*Clean bags approximated:* the test split has no clean bags, so each test image's empty content "
    f"(boxes that overlap no real object) stands in for one clean bag. Real clean bags may raise fewer "
    f"false alarms, so treat the clean row as an estimate."
)


# ----------------------------------------------------------------------- thresholds
def decide(score: float, t_low: float, t_high: float) -> str:
    """Bag decision from its highest box: pass below t_low, review up to t_high, search from t_high."""
    if score < t_low:
        return "pass"
    return "review" if score < t_high else "search"


def keep_low(t_low: float, t_high: float) -> tuple[float, float]:
    """The user moved t_low: push t_high up if needed so t_low <= t_high."""
    return t_low, max(t_low, t_high)


def keep_high(t_low: float, t_high: float) -> tuple[float, float]:
    """The user moved t_high: push t_low down if needed so t_low <= t_high."""
    return min(t_low, t_high), t_high


# ------------------------------------------------------------- cost and review load
def cost_summary(t_low: float, t_high: float) -> tuple[str, str]:
    """Headline numbers and the bag-level confusion matrix (per 1,000 bags) for one pair of thresholds."""
    r = costing.evaluate(SCORES, t_low, t_high)
    chosen = costing.evaluate(SCORES, CHOSEN_T_LOW, CHOSEN_T_HIGH)["cost_per_1000"]
    alternative = costing.evaluate(SCORES, ALTERNATIVE_T)["cost_per_1000"]
    band = "no review band" if t_low == t_high else f"review band {t_low:g} to {t_high:g}"
    diff = r["cost_per_1000"] - chosen

    headline = (
        f"### Thresholds {t_low:g} / {t_high:g} ({band})\n\n"
        f"| Expected cost per 1,000 bags | Sent to human review | Sent to manual search | Reviews a day "
        f"| Review shifts a day | Searches a day | Search shifts a day |\n"
        f"|---:|---:|---:|---:|---:|---:|---:|\n"
        f"| **€{r['cost_per_1000']:,.0f}** | **{100 * r['review_share']:.1f}%** | {100 * r['search_share']:.1f}% "
        f"| {r['reviews_per_day']:,.0f} | {r['shifts_per_day']:.1f} "
        f"| {r['searches_per_day']:,.0f} | {r['search_shifts_per_day']:.1f} |\n\n"
        f"Chosen policy ({CHOSEN}): €{chosen:,.0f} per 1,000 bags. "
        f"Alternative, one threshold at {ALTERNATIVE_T:g}: €{alternative:,.0f}. "
        f"Difference from the chosen policy at the current thresholds: {'+' if diff >= 0 else '-'}€{abs(diff):,.0f}."
    )

    per, n = r["per_1000_bags"], r["test_counts"]
    rows = [("Threat bag", "threat", 3), ("Clean bag", "clean", 1)]
    matrix = (
        "### Confusion matrix, bag level, per 1,000 bags\n\n"
        "Rows: what the bag really holds. Columns: what happens to it. "
        "In brackets: the test-split bags behind each share (156 threat bags, 156 clean stand-ins).\n\n"
        "| | Pass | Human review | Manual search |\n|---|---:|---:|---:|\n"
        + "".join(
            f"| **{label}** | "
            + " | ".join(f"{per[kind][o]:,.{digits}f} ({n[kind][o]})" for o in DECISIONS)
            + " |\n"
            for label, kind, digits in rows
        )
        + "\nA threat bag that passes is a missed threat. A clean bag sent to search is a false alarm."
    )
    return headline, matrix


def cost_curve(t_low: float, t_high: float):
    """Model-alone cost against threshold, with the chosen review band, the alternative and the sliders marked."""
    ink, ink2, muted, grid, axis, surface = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    series, current = "#2a78d6", "#d6532a"

    def at(t: float) -> float:
        return float(costing.cost_per_1000(costing.outcomes(SCORES, t), K))

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=110)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.step(costing.THRESHOLDS, CURVE, where="post", color=series, linewidth=2, label="Model alone (one threshold)")
    chosen_cost = costing.evaluate(SCORES, CHOSEN_T_LOW, CHOSEN_T_HIGH)["cost_per_1000"]
    ax.axvspan(CHOSEN_T_LOW, CHOSEN_T_HIGH, color=series, alpha=0.08, linewidth=0, zorder=0,
               label=f"Chosen policy: review band {CHOSEN}, €{chosen_cost:,.0f}")
    ax.plot([ALTERNATIVE_T], [at(ALTERNATIVE_T)], "o", markersize=15, markerfacecolor="none", markeredgecolor=series,
            markeredgewidth=2, zorder=3, label=f"Alternative: one threshold {ALTERNATIVE_T:g}: €{at(ALTERNATIVE_T):,.0f}")
    ax.plot([t_low], [at(t_low)], "D", markersize=7, color=current, markeredgecolor=surface,
            markeredgewidth=1.5, zorder=4, label=f"Slider t_low {t_low:g}, used alone: €{at(t_low):,.0f}")
    if t_low != t_high:
        band_cost = costing.evaluate(SCORES, t_low, t_high)["cost_per_1000"]
        ax.axhline(band_cost, color=current, linestyle="--", linewidth=1.2,
                   label=f"With review band {t_low:g} to {t_high:g}: €{band_cost:,.0f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(costing.THRESHOLDS[0] * 0.9, 1.0)
    ax.set_ylim(min(CURVE.min() * 0.3, 1e6), 1.6e6)
    ax.xaxis.set_major_locator(FixedLocator([0.001, 0.01, 0.1, 0.15, 0.7, 1.0]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_locator(FixedLocator([100, 1e3, 1e4, 1e5, 1e6]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "€1M" if v >= 1e6 else f"€{v:,.0f}"))
    ax.set_xlabel("Confidence threshold (log scale). A bag is searched if its highest box is at or above it.",
                  color=ink2, fontsize=9)
    ax.set_ylabel("Expected cost per 1,000 bags (log scale)", color=ink2, fontsize=9)
    ax.set_title("Expected cost against threshold (156 test bags, base rate 0.1%)", loc="left",
                 color=ink, fontsize=11, fontweight="bold")
    ax.grid(True, which="major", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(axis)
    ax.tick_params(colors=muted, labelsize=8.5, length=0, pad=5)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False, labelcolor=ink2)
    fig.tight_layout()
    plt.close(fig)
    return fig


def dashboard(t_low: float, t_high: float):
    """Everything on the cost tab for one pair of thresholds."""
    headline, matrix = cost_summary(t_low, t_high)
    return headline, matrix, cost_curve(t_low, t_high)


# --------------------------------------------------------------------- screen a bag
def rule(t_low: float, t_high: float) -> str:
    if t_low == t_high:
        return f"Rule: pass below {t_low:g}, manual search from {t_high:g} (no review band)."
    return f"Rule: pass below {t_low:g}, human review from {t_low:g} to {t_high:g}, manual search from {t_high:g}."


def screen(image, t_low: float, t_high: float):
    """Boxes at or above t_low drawn on the image, a box table, and the bag decision."""
    if image is None:
        return None, pd.DataFrame(columns=["Class", "Confidence", "Box goes to"]), "Upload an X-ray image."
    result = MODEL.predict(image, imgsz=IMGSZ, conf=PRED_CONF, verbose=False)[0]
    conf = result.boxes.conf.cpu().numpy()
    score = float(conf.max()) if len(conf) else 0.0  # bag score: highest box, any class (as in costing.py)

    shown = result[result.boxes.conf >= t_low]  # the boxes that matter at the current lower threshold
    annotated = shown.plot()[..., ::-1]  # Ultralytics draws in BGR; Gradio wants RGB
    table = pd.DataFrame(
        [
            (MODEL.names[int(c)], round(float(p), 3), DECISIONS[decide(float(p), t_low, t_high)])
            for c, p in zip(shown.boxes.cls.tolist(), shown.boxes.conf.tolist())
        ],
        columns=["Class", "Confidence", "Box goes to"],
    )

    decision = decide(score, t_low, t_high)
    top = f"{MODEL.names[int(result.boxes.cls[conf.argmax()])]} at {score:.3f}" if len(conf) else "no box at all"
    verdict = (
        f"## Bag decision: {DECISIONS[decision]}\n\n"
        f"Highest box: **{top}**. {rule(t_low, t_high)} "
        f"{len(shown)} box(es) at or above {t_low:g} are drawn."
    )
    return annotated, table, verdict


# ------------------------------------------------------------------ per-class results
def per_class_table() -> pd.DataFrame:
    """One row per model and class, plus an 'All classes' row with mAP, from artifacts/metrics.json."""
    names = {
        "finetuned_yolo11n": "Finetuned yolo11n",
        "coco_yolo11n_zero_shot": "COCO yolo11n, zero-shot (baseline)",
        "predict_nothing": "Predict nothing (baseline)",
    }

    def fmt(v):
        return "n/a" if v is None else f"{v:.3f}"

    rows = []
    for key, label in names.items():
        m = METRICS[key]
        for cls, c in m["per_class"].items():
            rows.append((label, cls, c["instances"], fmt(c["ap50"]), fmt(c["ap50_95"]),
                         fmt(c["precision@0.25"]), fmt(c["recall@0.25"])))
        total = sum(c["instances"] for c in m["per_class"].values())
        rows.append((label, "All classes (mAP)", total, fmt(m["map50"]), fmt(m["map50_95"]), "", ""))
    return pd.DataFrame(rows, columns=["Model", "Class", "Instances", "AP50", "AP50-95", "P", "R"])


# ------------------------------------------------------------------------------ UI
with gr.Blocks(title="X-ray baggage screening", analytics_enabled=False) as demo:
    gr.Markdown(
        "# X-ray baggage screening: thresholds, cost and review load\n"
        "Finetuned yolo11n on X-ray baggage (Gun, Knife, Pliers, Scissors, Wrench). "
        "A bag's score is its highest box, of any class. "
        f"**Chosen policy: {CHOSEN}.** Bags scoring below {CHOSEN_T_LOW:g} pass, "
        f"from {CHOSEN_T_LOW:g} up to {CHOSEN_T_HIGH:g} a screener checks the X-ray image and decides, "
        f"and from {CHOSEN_T_HIGH:g} up the bag goes straight to a manual search. "
        f"The alternative compared is one threshold at {ALTERNATIVE_T:g} (no review band). "
        "Move the sliders to see the cost and staffing of any pair."
    )
    with gr.Row():
        t_low = gr.Slider(0.001, 0.99, value=CHOSEN_T_LOW, step=0.001,
                          label="t_low: below this the bag passes")
        t_high = gr.Slider(0.001, 0.99, value=CHOSEN_T_HIGH, step=0.001,
                           label="t_high: at or above this the bag goes to a manual search (between: human review)")
    with gr.Row():
        chosen_btn = gr.Button(f"Chosen policy ({CHOSEN})", variant="primary")
        alternative_btn = gr.Button(f"Alternative: one threshold ({ALTERNATIVE_T:g})")

    with gr.Tabs():
        with gr.Tab("1. Cost and review load"):
            headline = gr.Markdown()
            with gr.Row():
                with gr.Column(scale=2):
                    curve = gr.Plot(label="Cost against threshold", show_label=False)
                with gr.Column(scale=1):
                    matrix = gr.Markdown()
            gr.Markdown(ASSUMPTIONS)

        with gr.Tab("2. Screen a bag"):
            gr.Markdown("Upload an X-ray image. The model runs once per change; "
                        "boxes at or above t_low are drawn and the bag decision uses both thresholds.")
            with gr.Row():
                image_in = gr.Image(type="pil", label="X-ray image")
                image_out = gr.Image(type="numpy", label="Predictions at the current threshold")
            verdict = gr.Markdown("Upload an X-ray image.")
            boxes = gr.Dataframe(label="Boxes at or above t_low", interactive=False)

        with gr.Tab("3. Per-class results"):
            gr.Markdown(
                "Test split (156 images, 358 objects). AP50 and AP50-95 per class; "
                "P and R at confidence 0.25 and IoU 0.5. The finetuned model against both baselines. "
                "From `artifacts/metrics.json`, written by `train.py`."
            )
            gr.Dataframe(per_class_table(), interactive=False, wrap=True)
            with gr.Row():
                gr.Image(str(ARTIFACTS / "ultralytics_test" / "confusion_matrix.png"),
                         label="Finetuned model: per-class confusion matrix (Ultralytics, test split)")
                gr.Image(str(ARTIFACTS / "ultralytics_test" / "BoxPR_curve.png"),
                         label="Finetuned model: precision-recall curve per class (Ultralytics, test split)")

    # Wiring: moving one slider keeps t_low <= t_high, then refreshes every view.
    views = [headline, matrix, curve]
    bag = [image_out, boxes, verdict]

    def refresh(lo, hi, image):
        return (lo, hi, *dashboard(lo, hi), *screen(image, lo, hi))

    outputs = [t_low, t_high, *views, *bag]
    # .input fires on dragging or typing (never on the programmatic updates below), so there is no loop.
    t_low.input(lambda lo, hi, im: refresh(*keep_low(lo, hi), im), [t_low, t_high, image_in], outputs,
                trigger_mode="always_last")
    t_high.input(lambda lo, hi, im: refresh(*keep_high(lo, hi), im), [t_low, t_high, image_in], outputs,
                 trigger_mode="always_last")
    chosen_btn.click(lambda im: refresh(CHOSEN_T_LOW, CHOSEN_T_HIGH, im), [image_in], outputs)
    alternative_btn.click(lambda im: refresh(ALTERNATIVE_T, ALTERNATIVE_T, im), [image_in], outputs)
    image_in.change(screen, [image_in, t_low, t_high], bag)
    demo.load(dashboard, [t_low, t_high], views)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=PORT)
