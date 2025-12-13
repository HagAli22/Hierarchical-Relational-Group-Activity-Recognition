تمام 👍
ده **README.md رسمي، منسّق، وجاهز للتحط مباشرة في الريبو**
(بصيغة Markdown نظيفة + عناوين + نقاط واضحة + مناسب لأي GitHub repo).

انسخه كما هو 👇

---

```markdown
# Hierarchical Relational Networks (HRN)
### Group Activity Recognition — Re-Implementation

This repository provides a **clean and faithful re-implementation** of the paper:

> **Hierarchical Relational Networks for Group Activity Recognition and Retrieval**  
> Mostafa S. Ibrahim, Greg Mori — ECCV 2018

The goal is to **understand, reproduce, and extend** the proposed architecture with a clear,
stage-wise implementation.

---

## Table of Contents
1. [Problem Definition](#problem-definition)
2. [Overall Pipeline](#overall-pipeline)
3. [Stage 1 — Person Feature Extraction](#stage-1--person-feature-extraction)
4. [Stage 2 — Relational Modeling (Single Frame)](#stage-2--relational-modeling-single-frame)
5. [Baseline Models (Non-Temporal)](#baseline-models-non-temporal)
6. [Scene Pooling](#scene-pooling)
7. [Stage 3 — Temporal Modeling (Full HRN)](#stage-3--temporal-modeling-full-hrn)
8. [Labels & Supervision](#labels--supervision)
9. [Dataset Structure](#dataset-structure)
10. [Training Strategy](#training-strategy)
11. [What This Repository Implements](#what-this-repository-implements)
12. [Design Notes](#design-notes)
13. [Future Extensions](#future-extensions)
14. [Reference](#reference)

---

## Problem Definition

The task is **Group Activity Recognition** from video clips.

Each clip:
- Contains **9 frames**
- Each frame has **multiple people (≈ 12 players)**
- Each person is annotated with a bounding box
- Each clip has **one group activity label**, e.g.:
  - `l-pass`, `r-pass`
  - `l-spike`, `r-spike`
  - `l-set`, `r-winpoint`

⚠️ **There is no frame-level classification.**  
✔️ The final prediction is always **clip-level**.

---

## Overall Pipeline

The model follows a **three-stage hierarchical pipeline**:

```

Stage 1: Person Feature Extraction
Stage 2: Relational Modeling (per frame)
Stage 3: Temporal Modeling (per clip)

```

High-level flow:

```

Frame:
Person crops → VGG19(fc7) → Relational Layers → Scene Feature (S_t)

Clip:
[S_1, S_2, ..., S_9] → LSTM → Group Activity

```

---

## Stage 1 — Person Feature Extraction

### Objective
Extract a strong appearance-based representation for each person.

### Details
- Backbone: **VGG19 (ImageNet pretrained)**
- Input: person bounding box crops
- Output: **fc7 feature (4096-D)**

### Training
- VGG19 is **fine-tuned** on person action labels
- Optimizer: Adam
- Learning rate: `1e-4`
- Batch size: `64`
- Epochs: `~200` (as in paper)

### Output Storage
Features are saved to **HDF5** for efficient reuse:
- `fc7_features`: `(N, 4096)`
- `labels`: person labels (optional)
- `meta`: `(video_id, clip_id, frame_id, box_id)`

This enables training later stages without reloading images.

---

## Stage 2 — Relational Modeling (Single Frame)

### Motivation
People in a scene interact — modeling them independently is suboptimal.

### Input
For a single frame:
```

P⁰ ∈ ℝ^{K × 4096}   (K players)

```

### Relational Layers
- Each layer models **inter-person relationships**
- Implemented using **shared MLPs over graph edges**
- Outputs updated person representations:
```

Pᴸ ∈ ℝ^{K × 128}

```

---

## Baseline Models (Non-Temporal)

These models operate on **a single frame only**  
(no temporal modeling, no LSTM).

### B1 — NoRelations
- fc7 → shared dense layer (4096 → 128)
- No relational reasoning
- Pool across players → scene feature
- Softmax → **Group Activity**

### RCRG Variants
Relational versions with different graph structures:

| Model | Description |
|------|-------------|
| RCRG-1R-1C | 1 relational layer, 1 clique (all players connected) |
| RCRG-1R-1C-!tuned | Same as above, without VGG fine-tuning |
| RCRG-2R-11C | 2 relational layers, both global |
| RCRG-2R-21C | Team-level → global relational layers |
| RCRG-3R-421C | Hierarchical cliques (4 → 2 → 1) |
| `-conc` | Uses concatenation pooling instead of max pooling |

These are used for **ablation studies**.

---

## Scene Pooling

After relational layers, person features are aggregated **within one frame**.

### Common Strategies
- **Max pooling** (default)
- Average pooling
- Concatenation pooling (`-conc`)
- Team-wise pooling (left/right team, then concatenate)

Result:
```

Scene Feature S_t ∈ ℝ^{128}

```

⚠️ Scene feature represents **one frame only**, not the clip.

---

## Stage 3 — Temporal Modeling (Full HRN)

This is the **final model** in the paper.

### Input
Sequence of scene features:
```

[S_1, S_2, ..., S_9]

```

### Model
- **LSTM**
- Takes ordered scene representations
- Final hidden state → Softmax

### Output
```

Group Activity (clip-level)

```

📌 Only this model uses **temporal information**.

---

## Labels & Supervision

- One label **per clip**
- No frame-level supervision
- Loss is applied only at the clip level

```

Clip → Group Activity

```

---

## Dataset Structure

### Stage 1 (Person-Level)
```

(image_crop, person_label)

```

### Stage 2 & 3 (Clip-Level)
```

clip_features: (K_max, T=9, 4096)
mask: (K_max, T)
label: group_activity

````

- `K_max` achieved via padding
- Players are ordered by bounding box `(x, y)` as in the paper

---

## Training Strategy

- **Two-stage training** (memory efficient):
  1. Train VGG19 → extract and save fc7
  2. Train relational + temporal models on fc7

- Optimizer: Adam
- Learning rate: `1e-4`
- Epochs: `200`
- Dropout: `50%` after relational input

---

## What This Repository Implements

✔️ Person feature extraction (VGG19)  
✔️ Baselines (B1, RCRG variants)  
✔️ Relational layers & clique structures  
✔️ Scene pooling strategies  
✔️ Temporal HRN with LSTM  
✔️ HDF5-based feature storage  
✔️ Clean dataset abstractions  

---

## Design Notes

- **Scene ≠ Group**
  - Scene feature = representation
  - Group activity = classification target
- Scene pooling is **not a learnable model**
- Relational layers are the core contribution
- LSTM is used only in the final HRN

---

## Future Extensions

- Attention-based pooling
- Graph Attention Networks (GAT)
- Learnable clique assignment
- Transformer-based temporal modeling

---

## Reference

```bibtex
@inproceedings{ibrahim2018hierarchical,
  title={Hierarchical relational networks for group activity recognition and retrieval},
  author={Ibrahim, Mostafa S and Mori, Greg},
  booktitle={ECCV},
  year={2018}
}
````

```

---

لو تحب:
- أعمل **README مختصر + README موسّع**
- أو أضيف **Implementation Checklist (TODO.md)**
- أو أربطه مباشرة بالكود اللي عندك (`src/`, `datasets/`, `models/`)

قولّي ✨
```
