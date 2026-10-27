"""
Run weights/best.pt on the validation split and save every box, for costing.py.

    uv run python predict_split.py           # -> artifacts/val_predictions.json
    uv run python predict_split.py test      # -> artifacts/test_predictions_finetuned.json

Same format and settings as artifacts/test_predictions.json (conf 0.001, the
image size recorded in artifacts/metrics.json). train.py calls the same function
after training, so this script is only for re-running it without training.
Runs fine on CPU (a minute or two for 312 images).
"""

import json
import sys

import train

if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "valid"
    settings = json.loads((train.ARTIFACTS_DIR / "metrics.json").read_text()).get("settings", {})
    _, names = train.write_data_yaml(train.DATA_DIR)
    out = None if split == "valid" else train.ARTIFACTS_DIR / f"{split}_predictions_finetuned.json"
    train.predict_split(train.WEIGHTS_DIR / "best.pt", train.DATA_DIR, names, split,
                        imgsz=int(settings.get("imgsz", train.IMGSZ)), out=out)
