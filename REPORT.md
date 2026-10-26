# Assignment 2 Report

- **Name**: Juan Felipe Ortiz Lejarde
- **Student ID**: 20417
- **Email**: jortiz.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Business problem, and the cost of being wrong

Every day, airport security staff spend their shifts looking at X-ray images
of bags and deciding, one by one, whether something inside could be dangerous.
People make mistakes, even more so after hours in front of a screen. The model
supports that same decision. Whether a bag is opened and searched by hand, or
allowed to go through.

- **Who acts on it**: the security screener at the checkpoint. The model flags
  the bag and draws a box around the object. The screener then clears the bag,
  looks again, or sends it to a manual search.
- **Positive class**: a bag that contains at least one of the five objects in
  the dataset (gun, knife, pliers, scissors or wrench). A clean bag is the
  negative.
- **False positive (a clean bag is flagged)**: the bag is searched for nothing.
  That means a few minutes of a screener's time and a delayed passenger. If it
  happens often, the whole queue slows down.
- **False negative (a threat goes through)**: a gun or a knife reaches the
  airside or the aircraft. If someone carries it with harmful intent, it can
  cost lives. Even when nobody is hurt, finding it later means evacuating a
  terminal, delaying or cancelling flights, and damaging the airport's
  reputation. Which can easily cost thousands, or millions, of euros.

The gap between the two is enormous. A false alarm costs a couple of euros of
time, whereas a missed weapon costs at least a million euros and possibly lives.  
Every later choice in this report follows from it: I care most about recall on the dangerous classes, I would
rather send a bag to a person than let it pass, and the threshold has to be
low.

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

Each image is an X-ray scan of a bag, similar to what a screener sees at the
checkpoint, and every object that should stop the bag is boxed and named. That
is why it serves the decision instead of only resembling it. The model has to
do the screener's job: find the object in the scan and say what it is.

| Class | Instances (all splits) | Share of instances | Test instances | Median box area (test) | Size scale |
|---|---:|---:|---:|---:|---|
| Gun | 1,751 | 22.5% | 81 | 3.3% | Medium |
| Knife | 1,594 | 20.5% | 88 | 8.2% | Large |
| Pliers | 1,542 | 19.8% | 67 | 2.8% | Small |
| Scissors | 1,369 | 17.6% | 54 | 1.9% | Small |
| Wrench | 1,518 | 19.5% | 68 | 5.4% | Medium |



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



- **The classes are balanced** (17.6% to 22.5% of instances each). No class is
  starved in training, and no single large class carries the average.
- **The test split has no clean bags.** All 156 test images contain at least
  one object. Real traffic is the opposite, since almost every bag is clean.
  Thus, this split cannot tell me how often the model raises a false alarm on
  a clean bag, which is precisely the number behind the cost of false
  positives. The only false-alarm signal left is boxes drawn on the background
  of bags that do contain something. -> Huge limitation
- **The test split is small.** There are only 48 to 53 test images per class,
  so one image more or less moves a class's recall by about 2 percentage
  points. Per-class differences of a few points are noise, not findings.
- **Scissors are the hardest objects to see.** They are the smallest class, and 15.4% of all scissor boxes cover less
  than 1% of the image. I expect them to be missed more often than the rest.
- **The train set is augmented 3x.** Its 3,303 images come from about 1,101
  source images, so the model has seen far less real variety than the count
  suggests.
- **What I cannot conclude**: the results show how well the model finds these
  five objects in bags from this dataset. Not how it behaves on real traffic,
  on other scanners, or on clean bags. (See next steps)


## Finetuning and baselines

**Settings** (all set in `train.py`):

- **Model**: `yolo11n` detection, the smallest YOLO11 model, finetuned from
  the COCO-pretrained checkpoint. It already knows general shapes and edges
  from COCO. Finetuning teaches it the five X-ray classes on top of that.
- **Image size** (`imgsz=416`): every image enters the model at 416x416
  pixels, the same size the dataset is exported at.
- **Epochs** (`epochs=50`): at most 50 full passes over the training images.
- **Batch** (`batch=16`): the model sees 16 images before each weight update.
- **Patience** (`patience=15`): training stops early if the validation score
  does not improve for 15 epochs in a row. It never triggered: all 50 epochs
  ran.
- **Seed** (`seed=0`): fixes the random choices (shuffling, augmentation), so
  a re-run starts from the same place.
- **Determinism** (`deterministic=True`): asks PyTorch for deterministic
  algorithms. Some GPU operations can still vary slightly, so a re-run gives
  close numbers, but not always identical ones.
- **Everything else** uses the Ultralytics 8.4 defaults.
- **Best checkpoint**: the epoch with the best validation fitness is copied to
  `weights/best.pt`.
- **Evaluation**: all three models are scored with the same code on the same 156 test images.
  AP50 and mAP50-95 use every box with conf >= 0.001, whereas P/R and the
  confusion matrix use conf >= 0.25.
  - COCO `yolo11n` zero-shot: only the COCO classes that also exist here
    (knife, scissors), mapped to Knife and Scissors.
  - Trivial baseline: predicts nothing.
- **Where trained**: Google Colab, T4 GPU, about 26 minutes for all 50 epochs.
  See [notes/colab.md](notes/colab.md).

Results on the test split (156 images, 358 objects), from the first Colab run:

| Model | mAP50 | mAP50-95 |
|---|---:|---:|
| Finetuned `yolo11n` | 0.903 | 0.690 |
| COCO `yolo11n` zero-shot | 0.010 | 0.008 |
| Predict nothing | 0.000 | 0.000 |

The finetuned model beats both baselines by a wide margin. The COCO model found
2 of the 54 scissors and nothing else, not a single knife, even though both
classes exist in COCO. Predicting nothing scores zero, as expected. What
surprised me was the size of the gap with COCO. Knowing what a knife looks
like in a photo does not help at all on an X-ray, so for this problem
finetuning is not optional: without it, there is no usable detector.


## Per-class evaluation and failure modes

Finetuned `yolo11n` on the test split (first Colab run). P and R at
conf >= 0.25, and the confusion matrix at the same threshold: "Correct" is the
diagonal, "Missed" is a true object predicted as background, and "False alarms"
are boxes drawn on the background.

| Class | Test instances | AP50 | AP50-95 | Precision | Recall | Correct | Missed | False alarms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gun | 81 | 0.958 | 0.812 | 0.814 | 0.975 | 79 | 2 | 17 |
| Knife | 88 | 0.880 | 0.633 | 0.705 | 0.841 | 74 | 11 | 31 |
| Pliers | 67 | 0.906 | 0.684 | 0.848 | 0.836 | 55 | 10 | 10 |
| Scissors | 54 | 0.893 | 0.669 | 0.712 | 0.870 | 47 | 7 | 19 |
| Wrench | 68 | 0.877 | 0.652 | 0.720 | 0.868 | 60 | 7 | 18 |

- **Gun carries the headline number**, and luckily it is the class that
  matters most. It has the best AP50 (0.958) and recall (0.975). Only 2 of 81
  guns were missed.
- **Knife is where the model fails most.** It has the most false alarms (31),
  the most misses (11), and the lowest precision (0.705) and AP50-95 (0.633).
  Knives are the largest objects, but they are long and thin, so the box is
  easy to place badly. At a checkpoint, that would mean more bags searched for
  nothing, and 11 knives going through.
- **Small scissors get missed.** 7 of 54 were missed and their precision is
  low (0.712), as the dataset section predicted.
- **Confusions between classes are rare.** Almost every error is an object
  missed as background or a box on the background, not one tool taken for
  another. The few that happen are small: 3 knives read as wrenches, and
  2 pliers read as a gun or a wrench. Pliers also lose 10 of 67 to the
  background, a higher share than Knife (15% against 13%).
- **COCO already has knife and scissors**, but not gun, pliers or wrench. Even
  so, the COCO model found no knives and almost no scissors, because X-ray
  images look nothing like the photos COCO was trained on. In real life, an
  off-the-shelf detector is of no use for screening. The model has to be
  trained on X-ray images, ideally from the airport's own scanners.
- **These false alarms fall on bags that already contain a threat**, so they
  would not change the decision for those bags. In real traffic they would land
  on clean bags, and this test split cannot show how often.



## Cost matrix and operating point

The decision is made per bag. If the model draws any box above the threshold,
the bag is flagged.

| Outcome | Cost | Where it comes from |
|---|---:|---|
| False positive (clean bag searched) | EUR 2 | A manual bag search costs about $1.58 (roughly EUR 1.5) in time-related losses ([Jin & Johnson, 2025](https://ojs.library.okstate.edu/osu/index.php/CARI/article/view/10383)). I round it up to EUR 2 to also cover the screener's time. |
| False negative (threat passes) | EUR 1,000,000 | One security breach at Munich in 2018, with no weapon involved, evacuated a terminal, cancelled about 330 flights and stranded more than 32,000 passengers. The airport put the bill at more than EUR 1 million ([Aviation Pros, 2018](https://www.aviationpros.com/airports/news/12422868/munich-airport-chaos-cost-1-million-airport-director-says)). I use it as a floor: if the object is used to hurt someone, the US Department of Transportation values one statistical life at $13.7 million ([US DOT, 2024 base year](https://www.transportation.gov/office-policy/transportation-policy/revised-departmental-guidance-on-valuation-of-a-statistical-life-in-economic-analysis)). |
| True positive (threat found) | EUR 2 | The same manual search, but this time it is the system working as intended. |
| True negative (clean bag passes) | EUR 0 | Nothing happens. |

As for the screener's time, one TSA posting pays a Transportation Security
Officer between 18.06 and 25.18 dollars per hour at Nebraska airports
([USAJOBS, 2023](https://www.usajobs.gov/job/741094200)).  In Spain, I assume a loaded
cost of EUR 20 per hour (salary, airport supplements and employer social
security). At that rate, two minutes of a screener's time is about EUR 0.70,
which fits within the EUR 2 above.

**Base rate.** TSA found 6,678 firearms at its checkpoints in 2024, or 7.4 per
million passengers screened
([TSA, 2025](https://www.tsa.gov/news/press/releases/2025/01/15/tsa-intercepts-6678-firearms-airport-security-checkpoints-2024)).
That counts guns only. Knives, scissors and tools are far more common in hand
luggage. I therefore assume that 1 bag in 1,000 (0.1%) contains one of the five objects. 

**What the costs imply.** A missed threat costs 500,000 times a false alarm.
To put it in simpler terms: catching one more threat is worth up to 500,000
extra searches, and with a 0.1% base rate that points to a low threshold. 

**Sweep and operating point.**

- **How the sweep scores a bag**: the decision is whether to open the bag, so a
  threat bag takes the confidence of its highest box, of any class and anywhere
  in the image. A threat is only missed when no box reaches the threshold. For
  clean bags I use the boxes that do not overlap any real object (IoU of 0.1 or
  less with every labelled box).
- Cost against threshold curve: see the chart below. Below 0.15, the cost
  falls slowly as the threshold rises, because fewer clean bags are searched.
  Above 0.15 it jumps, since each threat bag the model stops flagging adds
  about EUR 6,400 per 1,000 bags. Far more than the EUR 2,000 of searching all
  1,000 bags by hand.
- Chosen threshold (minimum expected cost): 0.15. It is the highest threshold
  that still flags all 156 test threat bags, because the weakest of them
  scores 0.165. At 0.15 the model also flags 20 of the 156 clean stand-ins, so
  about 13% of all bags would go to a manual search. 
- Expected cost per 1,000 bags at that threshold: EUR 258. Almost all of it is
  searches, EUR 256 for clean bags flagged by mistake and EUR 2 for the threat
  that is found. Searching every bag by hand would cost EUR 2,000, and doing
  nothing EUR 1,000,000.

![Expected cost per 1,000 bags against the confidence threshold, lowest at 0.15](artifacts/cost_curve.png)


**The assumption that would most change the answer is the cost of a missed
threat.** I assumed EUR 1 million because a breach with no weapon already cost
that much at Munich. Roughly, the threshold only moves up if the expected cost
of a miss per bag (base rate x cost of a miss) falls to the level of a search.
With a 0.1% base rate, that would take a missed object costing less than about
EUR 2,000. With the EUR 1 million cost, it would take fewer than about 2
threats per million bags, which is below even the firearm-only rate. Exact
break-even point: the best threshold first rises above 0.15 when a missed
threat costs less than about EUR 4,000, or when the base rate falls below
about 4 threats per million bags (0.0004%). Going from 0.15 to 0.228 gives up
2 of the 156 test threat bags but saves only 4 of the 156 clean searches.
About 2 searches per threat, which is why the exact figure is roughly twice my
rough one.



## What a person still has to check

Given how sensitive the matter is, I first planned a low threshold and a
person for everything uncertain. After the sweep, I decided against it. My
policy is one threshold and no review band: a bag whose highest box scores
0.15 or above goes straight to a manual search, and every other bag passes.
In the three-outcome framing, both thresholds sit at 0.15. Which leaves the
review band empty.

| Highest box confidence in the bag | Outcome |
|---|---|
| Below 0.15 | Pass |
| 0.15 or above | Fail: straight to a manual search |

The review band lost because of where the threat bags fall. 22 of the 156
test threat bags score between 0.15 and 0.7, and a screener who catches 90% of
the threats in the images they review (my assumption, not sourced) would let
about 1 in 10 of them through. At EUR 1 million per miss, that outweighs by
far the EUR 1.75 a review saves against a search.

**Search load.** People still open every flagged bag. My reference airport is
Adolfo Suárez Madrid-Barajas, which handled 66,196,984 passengers in 2024
([Aena, 2025](https://www.aena.es/es/prensa/los-aeropuertos-del-grupo-aena-registran-mas-de-3694-millones-de---pasajeros-en-2024.html)).
That is about 181,000 passengers a day, of whom about half depart. If each
departing passenger carries one bag (my assumption), roughly 90,700 bags a day
go through screening.

- Share of bags searched: 12.9%.
- Searches a day (90,700 x 12.9%): about 11,700.
- Time per search: two minutes, the screener time I already assumed in the
  cost section (my assumption, not sourced).
- People needed: about 390 hours of searching a day, or about 49 eight-hour
  shifts. The 20-minute limit applies to image review, not to searching a
  bag, so I count the whole shift.
- Is it realistic: yes. About 49 shifts is roughly a third of the 142 shifts
  it takes for a person to look at every image. It does depend on the
  false-alarm rate on real clean bags, which this test split can only
  approximate.

**The alternative I considered: a review band up to 0.7.** My first proposal
had three outcomes: pass below the lower threshold, human review from there up
to 0.7, and a manual search from 0.7 up. With the lower threshold at 0.15, it
sends 22 of the 156 test threat bags to a screener instead of straight to a
search. A review costs EUR 0.25 (30 seconds at EUR 20 per hour, plus the
matching share of breaks) against EUR 2 for a search, but that saving does not
pay for the threats the screeners miss. The design costs EUR 14,170 per 1,000
bags, against EUR 258 for my final policy.

- Share of bags sent to review: 10.9%, or about 9,900 bags a day (90,700 x
  10.9%).
- Time per review: I assume 30 seconds for a second look at one image (not
  sourced).
- Screeners may review images for at most 20 minutes in a row, followed by a
  break of at least 10 minutes (Regulation (EU) 2015/1998, as summarised by
  [CASRA, 2018](https://casra.ch/wp-content/uploads/2023/07/Science_Security_CASRA_Newsletter19_July2018.pdf)).
  Thus, an 8-hour shift gives at most about 5.3 hours of image review.
- People needed ((bags to review x 30 s) / 5.3 hours per shift):
  about 82 hours of image review a day, or about 15 screener shifts. As a
  rule of thumb, every 1% of bags sent to review is
  about 907 bags and 7.6 hours of image time a day, or about 1.4 screener
  shifts.
- Why I dropped it: the band only pays off if a screener catches at least
  99.86% of the threats shown to them. Even at 99% the sweep keeps no band,
  and only at 100% does the best upper threshold become 0.916 (EUR 34 per
  1,000 bags). I have no source for a catch rate that high.

**Compared with the alternatives** (per 1,000 bags, 0.1% base rate):

| Option | What happens | Cost per 1,000 bags |
|---|---|---:|
| Do nothing | No screening: every threat passes | EUR 1,000,000 (1 threat x EUR 1 million) |
| A person checks everything | Today's process. 1,000 x 30 s is 8.3 hours of image time, about 12.5 paid hours with breaks. At Barajas, about 142 screener shifts a day | EUR 100,252: EUR 250 in staff time, plus EUR 100,000 for the 1 threat in 10 that screeners miss if they catch 90% (my assumption, which this dataset cannot measure) |
| Search every bag by hand | Every bag is opened, so nothing passes unchecked | EUR 2,000 (1,000 searches x EUR 2) |
| The model alone (my policy) | One threshold at 0.15: flagged bags are searched, the rest pass | EUR 258 (13% of bags searched, all 156 test threats caught) |
| Model + human review (the alternative I considered) | Pass, review or fail with 0.15 and 0.7 | EUR 14,170 (11% of bags reviewed, reviewer catching 90%) |



## Limitations & next steps

- **No clean bags in the test split.** I cannot measure false alarms on clean
  bags, and that number decides both the review load and the cost of false
  positives. Next step: a bigger dataset that also includes clean bags, in
  proportions closer to real traffic.
- **Zero misses on a small test set.** The model missed none of the 156 test
  threat bags, but the 95% upper bound on its miss rate is still about 1.9%.
  At that rate it would cost more than searching every bag by hand. Next
  step: a test set with at least about 1,723 threat bags, plus clean bags, to
  show that the miss rate is below 0.17%.
- **Too few threat images.** With about 1,100 source images and 48 to 53 test
  images per class, a single image moves a class's recall by about 2 points.
  Next step: a bigger dataset with more items that should not pass screening.
  A more reliable model, and per-class numbers I can actually trust.
- **The train set is 3x augmented.** The model has seen less real variety than
  3,303 images suggest, and more real scans would help more than more
  augmentation.
- **One dataset, one source of scans.** The results may not hold on another
  airport's scanners. Next step: test on X-ray images from a different source
  before trusting the numbers.
- **Newer models.** I used YOLO11. The Ultralytics documentation (checked on
  26 October 2026) lists YOLO12, an attention-centric community model, and
  YOLO26, its latest release and the one it recommends for new projects. It
  also lists YOLO27 as upcoming, with no weights available yet
  ([Ultralytics models](https://docs.ultralytics.com/models/)). Next step:
  finetune YOLO12 and YOLO26 on the same splits, compare them with this model,
  and add YOLO27 once it is released.


## Generative AI use disclosure

I used generative AI during this assignment, in three ways.
Firstly, for research, finding the sources behind the cost figures and the base
rate. Secondly, for parts of the code: setting up the environment, downloading
the dataset, computing the dataset tables, among others. Lastly, to turn my own
notes into the prose of this report and format it. The decisions are mine: the
framing of the problem, the choice of dataset, the cost of each error, the
thresholds, how much goes to a person, and the conclusions.

