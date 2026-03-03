# FLAICOL: Flip-Point-Led Augmentation for Imbalanced Code-Mixed Offensive Language Detection

This repository contains the implementation of **FLAICOL**, a flip-point-led augmentation framework for low-resource, code-mixed offensive language detection, as described in the accompanying anonymous ACL submission :contentReference[oaicite:1]{index=1}.

FLAICOL addresses severe class imbalance in code-mixed hate speech datasets by generating minimal boundary-crossing examples in embedding space and using them to augment the minority class.

---

## 1. Overview

Hate speech detection in code-mixed Indian languages (Tamil–English, Malayalam–English, Kannada–English) presents three key challenges:

- Script mixing (native + Romanized)
- Orthographic variation
- Severe minority-class imbalance

FLAICOL introduces a flip-point-based augmentation framework that:

1. Fine-tunes a Transformer classifier.
2. Identifies near-boundary examples.
3. Performs embedding-space search under a bias-homotopy continuation strategy.
4. Maps continuous embedding perturbations back to discrete tokens.
5. Validates flips under the original classifier.
6. Augments minority-class samples.
7. Retrains the classifier and evaluates performance gains.

---

## 2. Method Summary

FLAICOL adapts the flip-point and homotopy framework of Yousefzadeh & O’Leary (2020) to pretrained Transformer classifiers.

For an input predicted as class *i*:

- A target class *j* is selected (typically highest competing logit).
- A minimal bias perturbation is solved via constrained optimization.
- Bias is interpolated back toward original parameters.
- At each step, embedding-space search identifies small semantic perturbations.
- The resulting embedding is mapped to discrete tokens via nearest-neighbor lookup.
- Only validated boundary-crossing flips are retained.

The final augmented examples are appended to the training set.

---

## 3. Models Used

Two encoder backbones were evaluated:

- **XLM-RoBERTa-base**
- **MuRIL-base**

Both were fine-tuned as sequence classification models.

---

## 4. Dataset

We use the **DravidianCodeMix Benchmark** introduced by Chakravarthi et al. (2022) :contentReference[oaicite:2]{index=2}.

The dataset consists of YouTube comments labeled as:

- `Not-Offensive`
- `Offensive`

### Language Splits

| Language | Total | Not-Offensive | Offensive |
|----------|-------|--------------|-----------|
| Tamil–English | 42,133 | 31,808 | 10,325 |
| Malayalam–English | 18,403 | 17,697 | 706 |
| Kannada–English | 5,874 | 4,397 | 1,477 |

Severe imbalance (especially Malayalam) motivates targeted augmentation.

### Dataset Source

The DravidianCodeMix dataset is publicly available from:

- Workshop website:  
  https://dravidianlangtech.github.io/

- Dataset paper:  
  Chakravarthi et al., 2022 (see references in :contentReference[oaicite:3]{index=3})

Please download the dataset from the official source and place the CSV files locally before running experiments.

---

## 5. Experimental Pipeline

The implementation follows four main stages:

### 1️⃣ Fine-tuning
A pretrained Transformer is fine-tuned on the training split.

### 2️⃣ Flip Discovery
For selected examples:
- Extract encoder representation.
- Solve bias homotopy subproblem.
- Perform embedding-space search.
- Decode continuous solution to tokens.
- Validate label flip under original classifier.

### 3️⃣ Augmentation
Validated flips are appended to the minority class (Offensive).

### 4️⃣ Retraining
The model is retrained on the augmented dataset.

---

## 6. Baseline Comparison

We compare FLAICOL augmentation against:

- No augmentation (baseline)
- SMOTE-based minority oversampling

FLAICOL produces small but consistent improvements in:
- Macro-F1
- Minority-class (Offensive) F1

---

## 7. Results Summary

Across all three language splits:

- Macro-F1 improves modestly (~ +0.5 to +1.0 points)
- Minority-class F1 improves consistently
- XLM-R produces semantically faithful flips
- MuRIL produces more stylistically diverse but noisier flips

Full quantitative results are detailed in the accompanying manuscript :contentReference[oaicite:4]{index=4}.

---

## 8. Running the Code

Set environment variables:

```bash
MODEL_OVERRIDE="xlm-roberta-base" \
DATA_CSV="tamil_train.csv" \
LANGUAGE="tamil" \
python run_experiment.py
