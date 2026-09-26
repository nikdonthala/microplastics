# Datasets

This directory contains the data used by the MicroScan AI training pipeline.

## Directory layout

```
data/
├── README.md            <- this file
├── training/            <- labelled CSV datasets used by train_model.py
└── sample/              <- demo images that can be analysed from the UI
```

## Training dataset format

`backend/training/train_model.py` expects a **CSV file** with exactly these
columns (order is preserved for training; the column NAMES must match):

```
area,perimeter,width,height,aspect_ratio,circularity,solidity,extent,
equivalent_diameter,mean_R,mean_G,mean_B,label
```

The `label` column must contain one of:

| Label      | Meaning                                                        |
| ---------- | -------------------------------------------------------------- |
| `fiber`    | Long, thin particle (high aspect ratio, low circularity)       |
| `fragment` | Irregular / angular shard                                      |
| `bead`     | Compact, approximately spherical particle                      |
| `other`    | Non-target material (dust, mineral grains, organic matter,...) |

### Where do rows come from?

Each row describes ONE particle that was detected in a microscopic image. The
intended workflow for building a real dataset:

1. Capture microscopic images of filtered water samples (with known scale).
2. Run detection + feature extraction (`POST /api/analyze` returns exactly
   these feature values for every particle).
3. Export the per-particle features to CSV.
4. **Manually label** each row (by visual inspection, ideally cross-checked by
   more than one person).

### DEMO dataset

`training/generate_demo_data.py --dataset` writes
`training/demo_particles.csv`, a **synthetic** dataset produced from the same
shape formulas the pipeline uses. It exists so the full
train -> evaluate -> inference loop can be demonstrated without a real
labelled dataset.

**Important:** rows in the demo dataset are simulated values, NOT scientific
measurements of real microplastics. Any metrics obtained from it (accuracy,
F1, ...) demonstrate that the pipeline functions correctly; they must not be
reported as scientific results.

### Minimum data requirements

- At least 2 distinct labels (more is better).
- Roughly balanced classes are recommended; `class_weight="balanced"`
  compensates for moderate imbalance.
- More rows per class -> more stable test metrics. With very few samples per
  class the stratified train/test split will refuse to run (by design).
