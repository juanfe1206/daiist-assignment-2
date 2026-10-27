# Training on Colab

My laptop has no NVIDIA GPU, so the real training runs on Colab.

1. Runtime > Change runtime type > T4 GPU.
2. Add the Roboflow key as a Colab secret: click the key icon on the left, name it `ROBOFLOW_API_KEY` and turn notebook access on. The key never goes in a cell.
3. Get the repo into Colab. Either clone the fork once it is pushed, or upload a zip of the repo (without `.venv/` and `datasets/`) and unzip it.
4. Run these cells:

```python
# Option A: clone the fork
!git clone -b assignment-2 https://github.com/<github-user>/<fork-name>.git repo
%cd repo
```

```python
# Option B: uploaded zip
!unzip -q repo.zip -d repo
%cd repo
```

```python
# Colab brings its own GPU PyTorch: no uv sync here.
# Pin the same ultralytics as uv.lock so the weights load locally.
!pip install -q "ultralytics==8.4.173" "roboflow>=1.7"

import os
from google.colab import userdata
os.environ["ROBOFLOW_API_KEY"] = userdata.get("ROBOFLOW_API_KEY")
```

```python
!python main.py train
```

```python
!zip -qr outputs.zip weights/best.pt artifacts notes/dataset_stats.md
from google.colab import files
files.download("outputs.zip")
```

5. Unzip `outputs.zip` into the local repo so `weights/best.pt` and `artifacts/` land in place, then commit them.

The submitted model (yolo11s, 512 px, 70 epochs) was trained on a Kaggle T4 GPU with the same pipeline, after Colab's free GPU quota ran out. The Kaggle notebook only swaps the Colab secret for a Kaggle secret (`ROBOFLOW_API_KEY`).
