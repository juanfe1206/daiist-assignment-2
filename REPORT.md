# Assignment 2 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you. Write only what you
can actually explain.*

- **Name**: Juan Felipe Ortiz Lejarde
- **Student ID**: 20417
- **Email**: jortiz.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Business problem, and the cost of being wrong

> **TODO (student):** airport X-ray baggage screening, deciding whether a bag
> goes to manual search. Who acts on the output, what a false positive and a
> false negative each cost, and why one is worse.

*The decision this model supports, who or what acts on it, and what happens
when it's wrong in each direction. Be concrete about the asymmetry: which
mistake is worse, roughly by how much, and why. Everything downstream (your
threshold, how much you send to a person, which metric you care about) has to trace
back to this section.*

## Dataset

- **Roboflow Universe**: workspace `vladutc`, project `x-ray-baggage`, version **3**
  (2024-05-19): https://universe.roboflow.com/vladutc/x-ray-baggage/dataset/3
- **License**: CC BY 4.0. Used as published, not forked.
- **Task**: object detection with 5 classes: Gun, Knife, Pliers, Scissors, Wrench.
  Each image is a single X-ray scan of one bag.
- **Size**: 3,771 images. Train 3,303, valid 312, test 156.
- **Preprocessing**: auto-orient, resize to 416x416 (stretch).
- **Augmentation (train only)**: each source image was turned into 3 versions,
  with horizontal and vertical flips, 90-degree rotations, a 0-30% crop and a
  +/-15 degree rotation. Thus, the 3,303 train images come from about 1,100
  source images.
- Version 1 has the same split sizes, preprocessing and augmentation. I used
  v3, the most recent one.

> **TODO (student):** why this dataset serves the screening decision rather
> than just looking like it.

*Workspace / project / version on Roboflow Universe, and the link. What one
image represents, what the classes are, and why this dataset actually serves
the problem above rather than merely resembling it.*

## What the dataset's composition implies

Computed by `train.py` from the downloaded labels (the full tables are in
[notes/dataset_stats.md](notes/dataset_stats.md)).

| Split | Images | Images with no objects | Objects | Objects per image |
|---|---:|---:|---:|---:|
| train | 3303 | 1 | 6712 | 2.03 |
| valid | 312 | 0 | 704 | 2.26 |
| test | 156 | 0 | 358 | 2.29 |

| Class | Train | Valid | Test | Test images with class | Median box area (train) | Median box area (test) | Boxes < 1% of image (all splits) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gun | 1508 | 162 | 81 | 49 | 4.4% | 3.3% | 2.8% |
| Knife | 1347 | 159 | 88 | 53 | 12.8% | 8.2% | 0.0% |
| Pliers | 1351 | 124 | 67 | 52 | 4.4% | 2.8% | 3.0% |
| Scissors | 1188 | 127 | 54 | 48 | 2.3% | 1.9% | 15.4% |
| Wrench | 1318 | 132 | 68 | 50 | 7.1% | 5.4% | 2.6% |

Box area is the box width x height as a share of the image. Train counts
include the augmented copies.

> **TODO (student):** what this implies. Which per-class numbers are too thin
> to trust, whether the test split looks like real bags, and what you therefore
> cannot conclude.

*Instances per class, train/val/test sizes, typical object size relative to the
image. Then the interpretation: which per-class numbers will be too thin to
trust, whether the test split resembles the images you'd really see, and what
you therefore can't conclude from your results.*

## Finetuning and baselines

**Settings** (all set in `train.py`):

- Model: `yolo11n` detection, finetuned from the COCO-pretrained checkpoint.
- `imgsz=416`, `epochs=50`, `batch=16`, `patience=15`, `seed=0`,
  `deterministic=True`. Everything else uses the Ultralytics 8.4 defaults.
- The best checkpoint (by validation fitness) is copied to `weights/best.pt`.
- All three models are scored with the same code on the same 156 test images.
  AP50 and mAP50-95 use every box with conf >= 0.001, whereas P/R and the
  confusion matrix use conf >= 0.25.
  - COCO `yolo11n` zero-shot: only the COCO classes that also exist here
    (knife, scissors), mapped to Knife and Scissors.
  - Trivial baseline: predicts nothing.
- **Where trained**: TODO (Google Colab, T4 GPU, once it has run. See
  [notes/colab.md](notes/colab.md).)

> **TODO (student):** the results table from `artifacts/results_table.md`, and
> what changed against the baselines.

*Which model and task (detection or classification) and why. A results table on
the same test split: your finetuned model, the pretrained COCO model zero-shot,
and a trivial baseline. What changed, and whether the gap is as large as you
expected.*

## Per-class evaluation and failure modes

> **TODO (student)**

*Per-class AP50, or per-class precision/recall and the confusion matrix.
Which class carries the headline number, which one is dragging, and what the
model actually confuses with what. Show where it breaks, not just that it
works.*

## Cost matrix and operating point

> **TODO (student)**

*Your euro cost for each kind of error and where each figure came from. The
base rate you assumed. The cost-against-threshold sweep, and the operating
point you chose. State the assumption that would most change your answer if it
were wrong.*

## What a person still has to check

> **TODO (student)**

*Your two thresholds, and what share of cases they send to a person. How many
that is per day, how long one takes to check, and how many people that needs.
Is that realistic? Then your design priced against the three alternatives:
doing nothing, having a person check everything, and letting the model decide
on its own.*

## Limitations ## Limitations & next steps next steps

> **TODO (student)**

*Real limitations you found, and concretely how you'd address each one with
more time or data. Not generic hedging.*

## Generative AI use disclosure

> **TODO (student):** your own statement. For reference, this is what AI
> (Claude Code) has done so far: cloned the repo and set up the environment,
> downloaded the dataset and computed the composition tables, and wrote
> `train.py` (dataset download, finetuning, test-split evaluation and both
> baselines).

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*
