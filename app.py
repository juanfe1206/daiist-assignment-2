"""
Assignment 2: Gradio dashboard for X-ray baggage screening.

    uv run python main.py app      # then open http://127.0.0.1:7860

Nothing is trained here. It loads weights/best.pt, the test-split bag scores
(costing.py, from artifacts/test_predictions.json) and artifacts/metrics.json.
The chosen policy is costing.chosen_pair: the cost-optimal (t_low, t_high).
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

import costing

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("APP_PORT", "7860"))  # check.py reads the same variable
IMGSZ = 416  # as in train.py
DECISIONS = {"pass": "Pass", "review": "Screener", "search": "Manual search"}

MODEL = YOLO(str(ROOT / "weights" / "best.pt"))
SCORES = costing.load_bag_scores()
CURVE = costing.sweep(SCORES)  # one-threshold cost per 1,000 bags at every threshold
METRICS = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
K = costing.DEFAULT_COSTS
T_LOW, T_HIGH = costing.chosen_pair(SCORES, K)

ASSUMPTIONS = (
    "Missed threat: " + ", ".join(f"{c.lower()} €{K.miss_cost(c):,.0f}" for c in costing.SEVERITY)
    + f". Manual search €{K.fp:g}, screener check €{K.review:.2f}. "
    f"Base rate 1 threat bag in {1 / K.base_rate:,.0f}. Screener catches {100 * K.reviewer_catch:g}% of threats."
)


def decide(score: float, t_low: float, t_high: float) -> str:
    if score < t_low:
        return "pass"
    return "review" if score < t_high else "search"


def cost_curve(t_low: float, t_high: float):
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=110)
    ax.step(costing.THRESHOLDS, CURVE, where="post", color="#2a78d6", lw=2, label="One threshold")
    ax.axvspan(T_LOW, T_HIGH, color="#2a78d6", alpha=0.1, lw=0, label=f"Chosen screener band {T_LOW:g}–{T_HIGH:g}")
    for t in (t_low, t_high):
        ax.axvline(t, color="#d6532a", ls="--", lw=1.2)
    ax.plot([], [], color="#d6532a", ls="--", label="Sliders")
    ax.set(xscale="log", yscale="log", xlim=(0.001, 1),
           xlabel="Threshold", ylabel="Cost per 1,000 bags (€)")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    plt.close(fig)
    return fig


def update(t_low: float, t_high: float):
    """Cost, shares, confusion matrix and curve for one pair of thresholds."""
    r = costing.evaluate(SCORES, t_low, t_high, K)
    per = r["per_1000_bags"]
    matrix = pd.DataFrame(
        [[label] + [f"{per[kind][o]:,.{d}f}" for o in DECISIONS] for label, kind, d in
         [("Threat bag", "threat", 3), ("Clean bag", "clean", 1)]],
        columns=["Per 1,000 bags", *DECISIONS.values()],
    )
    return (f"€{r['cost_per_1000']:,.0f}", f"{100 * r['review_share']:.1f}%", f"{100 * r['search_share']:.1f}%",
            matrix, cost_curve(t_low, t_high))


def screen(image, t_low: float, t_high: float):
    """Boxes at or above t_low on the image, the bag decision, and a box table."""
    if image is None:
        return None, "", None
    res = MODEL.predict(image, imgsz=IMGSZ, conf=0.001, verbose=False)[0]
    score = float(res.boxes.conf.max()) if len(res.boxes) else 0.0  # bag score: highest box
    shown = res[res.boxes.conf >= t_low]
    boxes = pd.DataFrame(
        [(MODEL.names[int(c)], round(float(p), 3)) for c, p in zip(shown.boxes.cls, shown.boxes.conf)],
        columns=["Class", "Confidence"],
    )
    verdict = f"### {DECISIONS[decide(score, t_low, t_high)]} (highest box {score:.3f})"
    return shown.plot()[..., ::-1], verdict, boxes


def per_class() -> pd.DataFrame:
    m = METRICS["finetuned_yolo11n"]
    rows = [(c, v["ap50"], v["precision@0.25"], v["recall@0.25"]) for c, v in m["per_class"].items()]
    rows.append(("All (mAP50)", m["map50"], None, None))
    return pd.DataFrame(rows, columns=["Class", "AP50", "Precision", "Recall"]).round(3)


def keep_low(lo, hi):
    return lo, max(lo, hi)


def keep_high(lo, hi):
    return min(lo, hi), hi


with gr.Blocks(title="X-ray baggage screening", analytics_enabled=False) as demo:
    gr.Markdown("# X-ray baggage screening\n"
                "A bag passes below t_low, a screener checks it between, and it is searched by hand from t_high.")
    with gr.Row():
        lo = gr.Slider(0.001, 0.99, value=T_LOW, step=0.001, label="t_low (pass below)")
        hi = gr.Slider(0.001, 0.99, value=T_HIGH, step=0.001, label="t_high (manual search from)")
    reset = gr.Button(f"Reset to chosen policy ({T_LOW:g} / {T_HIGH:g})", size="sm")
    with gr.Accordion("Assumptions", open=False):
        gr.Markdown(ASSUMPTIONS)

    with gr.Tabs():
        with gr.Tab("Cost"):
            with gr.Row():
                cost = gr.Textbox(label="Expected cost per 1,000 bags", interactive=False)
                review = gr.Textbox(label="Sent to screener", interactive=False)
                search = gr.Textbox(label="Sent to manual search", interactive=False)
            matrix = gr.Dataframe(label="Confusion matrix (rows: truth, columns: decision)", interactive=False)
            curve = gr.Plot(show_label=False)
        with gr.Tab("Upload a bag"):
            with gr.Row():
                image_in = gr.Image(type="pil", label="X-ray image")
                image_out = gr.Image(label="Boxes at or above t_low")
            verdict = gr.Markdown()
            boxes = gr.Dataframe(interactive=False, show_label=False)
        with gr.Tab("Per-class results"):
            gr.Dataframe(per_class(), label="Finetuned model, test split (precision and recall at confidence 0.25)",
                         interactive=False)
            gr.Markdown(f"Baselines, mAP50: COCO zero-shot {METRICS['coco_yolo11n_zero_shot']['map50']:.3f}, "
                        f"predict nothing {METRICS['predict_nothing']['map50']:.3f}.")

    views, bag = [cost, review, search, matrix, curve], [image_out, verdict, boxes]

    def refresh(t_low, t_high, image):
        return t_low, t_high, *update(t_low, t_high), *screen(image, t_low, t_high)

    outs = [lo, hi, *views, *bag]
    lo.input(lambda a, b, im: refresh(*keep_low(a, b), im), [lo, hi, image_in], outs, trigger_mode="always_last")
    hi.input(lambda a, b, im: refresh(*keep_high(a, b), im), [lo, hi, image_in], outs, trigger_mode="always_last")
    reset.click(lambda im: refresh(T_LOW, T_HIGH, im), [image_in], outs)
    image_in.change(screen, [image_in, lo, hi], bag)
    demo.load(update, [lo, hi], views)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=PORT)
