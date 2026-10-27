"""
Assignment 2 training pipeline: X-ray baggage threat detection.

    uv run python main.py train               # full pipeline
    uv run python train.py --stats-only       # download + dataset stats, no training

Steps:
1. Download Roboflow Universe dataset vladutc/x-ray-baggage (version below) in
   YOLO format into datasets/ (gitignored). The key comes from api_key.py.
2. Describe the dataset: images, empty images and instances per split, and
   bbox area relative to the image per class -> notes/dataset_stats.md.
3. Finetune yolo11n from the COCO-pretrained checkpoint and copy best.pt to
   weights/best.pt.
4. Evaluate on the test split, with the same code for all three models:
   the finetuned model, COCO yolo11n zero-shot (only COCO classes that exist in
   this dataset: knife, scissors), and a trivial "predict nothing" baseline.
   -> artifacts/metrics.json, artifacts/results_table.md,
      artifacts/test_predictions.json (every box with conf >= 0.001, plus the
      ground truth, so thresholds can be swept later without re-running models)
   The same boxes on the validation split -> artifacts/val_predictions.json
   (costing.py picks the thresholds on it and reports their cost on the test split).
   Ultralytics' own test-split validation of the finetuned model (with its
   confusion matrix and PR plots) is copied into artifacts/ultralytics_test/.
5. Cost sweep (costing.py): thresholds chosen on artifacts/val_predictions.json,
   cost per 1,000 bags reported on artifacts/test_predictions.json.
   -> artifacts/cost_sweep.json, artifacts/cost_curve.png

Training runs on CPU without a GPU but is very slow there. See notes/colab.md.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, box_iou

import costing

ROOT = Path(__file__).resolve().parent

# Dataset (Roboflow Universe, used as-is, not forked)
WORKSPACE = "vladutc"
PROJECT = "x-ray-baggage"
VERSION = 3
DATA_DIR = ROOT / "datasets" / f"{PROJECT}-v{VERSION}"

# Outputs
WEIGHTS_DIR = ROOT / "weights"
ARTIFACTS_DIR = ROOT / "artifacts"
NOTES_DIR = ROOT / "notes"
RUNS_DIR = ROOT / "runs"

# Model and training settings
BASE_MODEL = "yolo11n.pt"  # COCO-pretrained starting point for the finetune
ZERO_SHOT_MODEL = "yolo11n.pt"  # COCO-pretrained zero-shot baseline
IMGSZ = 416  # the dataset is exported at 416x416 (stretch)
TRAIN_ARGS = dict(epochs=50, batch=16, patience=15, seed=0, deterministic=True)
# Key for the finetuned model in metrics.json / *_predictions.json. Taken from
# costing.py so the cost sweep at the end always finds it (older runs wrote
# "finetuned_yolo11n"; costing.model_key accepts either).
FINETUNED = costing.MODEL

# Evaluation settings
PRED_CONF = 0.001  # keep almost every box; operating thresholds are chosen later
CM_CONF = 0.25  # confidence used for the confusion matrix and P/R tables
CM_IOU = 0.45  # IoU to count a detection as matching a ground-truth box (CM)
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)  # for AP50 and mAP50-95
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")


# --------------------------------------------------------------------------- data
def download_dataset() -> Path:
    from roboflow import Roboflow

    from api_key import load_api_key

    rf = Roboflow(api_key=load_api_key())
    version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
    version.download("yolov11", location=str(DATA_DIR))  # skips if already there
    return DATA_DIR


def write_data_yaml(data_dir: Path) -> tuple[Path, list[str]]:
    """Rewrite Roboflow's data.yaml with absolute paths so Ultralytics finds every split."""
    names = yaml.safe_load((data_dir / "data.yaml").read_text(encoding="utf-8"))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    cfg = {
        "path": str(data_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": dict(enumerate(names)),
    }
    out = data_dir / "data_abs.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out, list(names)


def split_images(data_dir: Path, split: str) -> list[Path]:
    folder = data_dir / split / "images"
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def read_labels(image_path: Path) -> np.ndarray:
    """YOLO labels for one image as (n, 5): class, cx, cy, w, h (normalised).

    Polygon rows (class x1 y1 x2 y2 ...) are turned into their bounding box.
    """
    label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    rows = []
    if label_path.is_file():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = [float(v) for v in line.split()]
            if len(values) == 5:
                rows.append(values)
            elif len(values) > 5:
                xs, ys = values[1::2], values[2::2]
                x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                rows.append([values[0], (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
    return np.array(rows, dtype=float).reshape(-1, 5)


def dataset_stats(data_dir: Path, names: list[str]) -> dict:
    """Counts per split and per class, and bbox area as a fraction of the image."""
    stats = {"splits": {}, "classes": {n: {} for n in names}}
    for split in SPLITS:
        images = split_images(data_dir, split)
        labels = [read_labels(p) for p in images]
        all_boxes = np.concatenate(labels) if labels else np.zeros((0, 5))
        stats["splits"][split] = {
            "images": len(images),
            "images_without_objects": sum(len(lab) == 0 for lab in labels),
            "objects": len(all_boxes),
        }
        for k, name in enumerate(names):
            boxes = all_boxes[all_boxes[:, 0] == k]
            area = boxes[:, 3] * boxes[:, 4]
            stats["classes"][name][split] = {
                "instances": len(boxes),
                "images_with_class": sum(bool((lab[:, 0] == k).any()) for lab in labels),
                "median_area_frac": float(np.median(area)) if len(area) else None,
                "share_below_1pct": float((area < 0.01).mean()) if len(area) else None,
            }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    write_stats_markdown(stats, names)
    return stats


def write_stats_markdown(stats: dict, names: list[str]) -> None:
    def pct(x):
        return "n/a" if x is None else f"{100 * x:.1f}%"

    lines = [
        f"# Dataset stats: {WORKSPACE}/{PROJECT} v{VERSION}",
        "",
        "Generated by `train.py` from the downloaded YOLO labels.",
        "The train split includes Roboflow's 3x augmented copies.",
        "Box area is the box width x height as a share of the image area.",
        "",
        "## Images per split",
        "",
        "| Split | Images | Images with no objects | Objects | Objects per image |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        s = stats["splits"][split]
        per_image = s["objects"] / s["images"] if s["images"] else 0
        lines.append(
            f"| {split} | {s['images']} | {s['images_without_objects']} | "
            f"{s['objects']} | {per_image:.2f} |"
        )
    lines += [
        "",
        "## Instances per class",
        "",
        "| Class | Train | Valid | Test | Test images with class |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in names:
        c = stats["classes"][name]
        lines.append(
            f"| {name} | {c['train']['instances']} | {c['valid']['instances']} | "
            f"{c['test']['instances']} | {c['test']['images_with_class']} |"
        )
    lines += [
        "",
        "## Object size relative to the image",
        "",
        "| Class | Median box area (train) | Median box area (test) | Boxes < 1% of image (all splits) |",
        "|---|---:|---:|---:|",
    ]
    for name in names:
        c = stats["classes"][name]
        n_all = sum(c[s]["instances"] for s in SPLITS)
        small = sum((c[s]["share_below_1pct"] or 0) * c[s]["instances"] for s in SPLITS)
        lines.append(
            f"| {name} | {pct(c['train']['median_area_frac'])} | "
            f"{pct(c['test']['median_area_frac'])} | {pct(small / n_all if n_all else None)} |"
        )
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    (NOTES_DIR / "dataset_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------- training
def finetune(data_yaml: Path, train_args: dict = TRAIN_ARGS) -> Path:
    model = YOLO(BASE_MODEL)
    model.train(
        data=str(data_yaml),
        imgsz=IMGSZ,
        project=str(RUNS_DIR),
        name="finetune",
        exist_ok=True,
        **train_args,
    )
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    best = WEIGHTS_DIR / "best.pt"
    shutil.copy2(model.trainer.best, best)
    print(f"Best weights copied to {best}")
    return best


# --------------------------------------------------------------------- evaluation
def predict(model: YOLO, images: list[Path], class_map: dict[int, int], imgsz: int = IMGSZ) -> list[np.ndarray]:
    """Per image (n, 6): class, conf, x1, y1, x2, y2 in pixels, mapped to dataset classes."""
    out = []
    for result in model.predict(
        source=[str(p) for p in images],
        imgsz=imgsz,
        conf=PRED_CONF,
        classes=sorted(class_map),
        stream=True,
        verbose=False,
    ):
        boxes = result.boxes
        cls = np.array([class_map[int(c)] for c in boxes.cls.tolist()], dtype=float)
        out.append(np.column_stack([cls, boxes.conf.cpu().numpy(), boxes.xyxy.cpu().numpy()]).reshape(-1, 6))
    return out


def ground_truth(images: list[Path]) -> list[np.ndarray]:
    """Per image (n, 5): class, x1, y1, x2, y2 in pixels."""
    import cv2

    out = []
    for path in images:
        h, w = cv2.imread(str(path)).shape[:2]
        lab = read_labels(path)
        cx, cy, bw, bh = lab[:, 1] * w, lab[:, 2] * h, lab[:, 3] * w, lab[:, 4] * h
        out.append(np.column_stack([lab[:, 0], cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]))
    return out


def iou_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return box_iou(torch.tensor(gt[:, 1:5]), torch.tensor(pred[:, 2:6])).numpy()


def match_tp(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """(n_pred, 10) bool: is each prediction a true positive at IoU 0.50:0.95 (same rule as Ultralytics)."""
    correct = np.zeros((len(pred), len(IOU_THRESHOLDS)), dtype=bool)
    if not len(gt) or not len(pred):
        return correct
    iou = iou_matrix(gt, pred) * (gt[:, :1] == pred[:, 0][None, :])
    for i, threshold in enumerate(IOU_THRESHOLDS):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if len(matches):
            matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1], i] = True
    return correct


def confusion_matrix(gts: list[np.ndarray], preds: list[np.ndarray], nc: int) -> np.ndarray:
    """(nc+1, nc+1) counts, rows = true class, columns = predicted class, last = background."""
    cm = np.zeros((nc + 1, nc + 1), dtype=int)
    for gt, pred in zip(gts, preds):
        pred = pred[pred[:, 1] >= CM_CONF]
        matched_gt, matched_pred = set(), set()
        if len(gt) and len(pred):
            iou = iou_matrix(gt, pred)
            pairs = np.array(np.nonzero(iou > CM_IOU)).T
            if len(pairs):
                pairs = pairs[iou[pairs[:, 0], pairs[:, 1]].argsort()[::-1]]
                for g, p in pairs:
                    if g not in matched_gt and p not in matched_pred:
                        cm[int(gt[g, 0]), int(pred[p, 0])] += 1
                        matched_gt.add(g)
                        matched_pred.add(p)
        for g in range(len(gt)):
            if g not in matched_gt:
                cm[int(gt[g, 0]), nc] += 1  # missed: true object, predicted background
        for p in range(len(pred)):
            if p not in matched_pred:
                cm[nc, int(pred[p, 0])] += 1  # false alarm on background
    return cm


def evaluate(gts: list[np.ndarray], preds: list[np.ndarray], names: list[str]) -> dict:
    nc = len(names)
    tp = np.concatenate([match_tp(g, p) for g, p in zip(gts, preds)]).reshape(-1, len(IOU_THRESHOLDS))
    pred_all = np.concatenate(preds).reshape(-1, 6)
    target_cls = np.concatenate([g[:, 0] for g in gts])

    ap = np.zeros((nc, len(IOU_THRESHOLDS)))
    if len(pred_all):
        result = ap_per_class(tp, pred_all[:, 1], pred_all[:, 0], target_cls)
        for row, k in enumerate(result[6]):
            ap[int(k)] = result[5][row]

    keep = pred_all[:, 1] >= CM_CONF  # P/R at the fixed confidence used for the confusion matrix
    per_class = {}
    for k, name in enumerate(names):
        n_true = int((target_cls == k).sum())
        sel = keep & (pred_all[:, 0] == k)
        n_pred, n_tp = int(sel.sum()), int(tp[sel, 0].sum())
        per_class[name] = {
            "instances": n_true,
            "ap50": float(ap[k, 0]),
            "ap50_95": float(ap[k].mean()),
            f"precision@{CM_CONF}": n_tp / n_pred if n_pred else None,
            f"recall@{CM_CONF}": n_tp / n_true if n_true else None,
        }
    present = [k for k in range(nc) if (target_cls == k).any()]
    return {
        "map50": float(ap[present, 0].mean()) if present else 0.0,
        "map50_95": float(ap[present].mean()) if present else 0.0,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": names + ["background"],
            "rows": "true",
            "columns": "predicted",
            "conf": CM_CONF,
            "iou": CM_IOU,
            "counts": confusion_matrix(gts, preds, nc).tolist(),
        },
    }


def ultralytics_test_val(weights: Path, data_yaml: Path) -> dict:
    """Ultralytics' own validation of the finetuned model on the test split (plots + cross-check)."""
    result = YOLO(str(weights)).val(
        data=str(data_yaml), split="test", imgsz=IMGSZ, project=str(RUNS_DIR), name="test_eval", exist_ok=True
    )
    dest = ARTIFACTS_DIR / "ultralytics_test"
    dest.mkdir(parents=True, exist_ok=True)
    for png in Path(result.save_dir).glob("*.png"):
        shutil.copy2(png, dest / png.name)
    return {"map50": float(result.box.map50), "map50_95": float(result.box.map)}


def evaluate_all(weights: Path, data_dir: Path, data_yaml: Path, names: list[str]) -> dict:
    images = split_images(data_dir, "test")
    gts = ground_truth(images)

    finetuned = YOLO(str(weights))
    coco = YOLO(ZERO_SHOT_MODEL)
    lower = {n.lower(): k for k, n in enumerate(names)}
    coco_map = {cid: lower[cname.lower()] for cid, cname in coco.names.items() if cname.lower() in lower}
    print(f"COCO classes used zero-shot: {[coco.names[c] for c in coco_map]}")

    preds = {
        FINETUNED: predict(finetuned, images, {k: k for k in range(len(names))}),
        "coco_yolo11n_zero_shot": predict(coco, images, coco_map),
        "predict_nothing": [np.zeros((0, 6)) for _ in images],
    }
    metrics = {name: evaluate(gts, p, names) for name, p in preds.items()}
    metrics[FINETUNED]["ultralytics_val_test"] = ultralytics_test_val(weights, data_yaml)
    metrics["settings"] = {
        "dataset": f"{WORKSPACE}/{PROJECT}/{VERSION}",
        "base_model": BASE_MODEL,
        "zero_shot_model": ZERO_SHOT_MODEL,
        "imgsz": IMGSZ,
        **TRAIN_ARGS,
        "pred_conf": PRED_CONF,
        "cm_conf": CM_CONF,
        "cm_iou": CM_IOU,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "test_images": len(images),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_predictions(ARTIFACTS_DIR / "test_predictions.json", names, images, gts, preds)
    write_results_table(metrics, names)
    return metrics


def write_predictions(path: Path, names: list[str], images: list[Path], gts: list[np.ndarray], preds: dict) -> None:
    """Every box (conf >= PRED_CONF) and the ground truth per image, one key per model."""
    path.write_text(json.dumps({
        "names": names,
        "columns": {"gt": "class,x1,y1,x2,y2 (px)", "pred": "class,conf,x1,y1,x2,y2 (px)"},
        "images": [
            {
                "file": img.name,
                "gt": np.round(g, 2).tolist(),
                **{m: np.round(p[i], 4).tolist() for m, p in preds.items()},
            }
            for i, (img, g) in enumerate(zip(images, gts))
        ],
    }))


def predict_split(weights: Path, data_dir: Path, names: list[str], split: str = "valid",
                  imgsz: int = IMGSZ, out: Path | None = None) -> Path:
    """The finetuned model's boxes on one split, in the test_predictions.json format.

    Default: the validation split -> artifacts/val_predictions.json, where costing.py picks
    its thresholds. Standalone: uv run python predict_split.py
    """
    images = split_images(data_dir, split)
    preds = {FINETUNED: predict(YOLO(str(weights)), images, {k: k for k in range(len(names))}, imgsz)}
    out = out or ARTIFACTS_DIR / ("val_predictions.json" if split == "valid" else f"{split}_predictions.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(out, names, images, ground_truth(images), preds)
    print(f"Wrote {out} ({len(images)} images, imgsz {imgsz})")
    return out


def write_results_table(metrics: dict, names: list[str]) -> None:
    models = [FINETUNED, "coco_yolo11n_zero_shot", "predict_nothing"]

    def fmt(x):
        return "n/a" if x is None else f"{x:.3f}"

    lines = [
        "# Test split results",
        "",
        f"P and R at conf >= {CM_CONF}, IoU >= 0.5. Generated by `train.py`.",
        "",
        "| Model | mAP50 | mAP50-95 |",
        "|---|---:|---:|",
    ]
    lines += [f"| {m} | {fmt(metrics[m]['map50'])} | {fmt(metrics[m]['map50_95'])} |" for m in models]
    for m in models:
        lines += ["", f"## {m}", "", "| Class | Instances | AP50 | AP50-95 | P | R |", "|---|---:|---:|---:|---:|---:|"]
        for name in names:
            c = metrics[m]["per_class"][name]
            lines.append(
                f"| {name} | {c['instances']} | {fmt(c['ap50'])} | {fmt(c['ap50_95'])} | "
                f"{fmt(c[f'precision@{CM_CONF}'])} | {fmt(c[f'recall@{CM_CONF}'])} |"
            )
        cm = metrics[m]["confusion_matrix"]
        lines += ["", "Confusion matrix (rows true, columns predicted):", "",
                  "| | " + " | ".join(cm["labels"]) + " |", "|---" * (len(cm["labels"]) + 1) + "|"]
        lines += [f"| {lab} | " + " | ".join(map(str, row)) + " |" for lab, row in zip(cm["labels"], cm["counts"])]
    text = "\n".join(lines) + "\n"
    (ARTIFACTS_DIR / "results_table.md").write_text(text, encoding="utf-8")
    print(text)


# --------------------------------------------------------------------------- main
def run(data_dir: Path, train_args: dict = TRAIN_ARGS) -> None:
    data_yaml, names = write_data_yaml(data_dir)
    dataset_stats(data_dir, names)
    weights = finetune(data_yaml, train_args)
    evaluate_all(weights, data_dir, data_yaml, names)
    predict_split(weights, data_dir, names, "valid")
    costing.run(ARTIFACTS_DIR / "test_predictions.json", ARTIFACTS_DIR)


if __name__ == "__main__":
    data_dir = download_dataset()
    if "--stats-only" in sys.argv:
        dataset_stats(data_dir, write_data_yaml(data_dir)[1])
        print((NOTES_DIR / "dataset_stats.md").read_text(encoding="utf-8"))
    else:
        run(data_dir)
