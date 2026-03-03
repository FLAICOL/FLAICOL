# FLAICOL: Flip-Point-Led Augmentation for Imbalanced Code-Mixed Offensive Language Detection

This repository contains the implementation of **FLAICOL**, a flip-point-led augmentation framework for low-resource, code-mixed offensive language detection, as described in the accompanying anonymous ACL submission.

FLAICOL addresses severe class imbalance in code-mixed hate speech datasets by generating minimal boundary-crossing examples in embedding space and using them to augment the minority class.

---

## 1. Overview

Hate speech detection in code-mixed Indian languages (Tamil–English, Malayalam–English, Kannada–English) presents three key challenges:

* Script mixing (native + Romanized)
* Orthographic variation
* Severe minority-class imbalance

FLAICOL introduces a flip-point-based augmentation framework that:

1.  Fine-tunes a Transformer classifier.
2.  Identifies near-boundary examples.
3.  Performs embedding-space search under a bias-homotopy continuation strategy.
4.  Maps continuous embedding perturbations back to discrete tokens.
5.  Validates flips under the original classifier.
6.  Augments minority-class samples.
7.  Retrains the classifier and evaluates performance gains.

---

## 2. Method Summary

FLAICOL adapts the flip-point and homotopy framework introduced by [Yousefzadeh & O’Leary](https://proceedings.mlr.press/v107/yousefzadeh20a.html) to pretrained Transformer classifiers.

For an input predicted as class *i*:

* A target class *j* is selected (typically highest competing logit).
* A minimal bias perturbation is solved via constrained optimization.
* Bias is interpolated back toward original parameters.
* At each step, embedding-space search identifies small semantic perturbations.
* The resulting embedding is mapped to discrete tokens via nearest-neighbor lookup.
* Only validated boundary-crossing flips are retained.

The final augmented examples are appended to the training set.

---

## 3. Models Used

Two encoder backbones were evaluated:

* **XLM-RoBERTa-base**
* **MuRIL-base**

Both were fine-tuned as sequence classification models.

---

## 4. Dataset

We use the **DravidianCodeMix Benchmark** introduced by Chakravarthi et al. (2022).

The final dataset format consists of YouTube comments labeled as:

* `Not-Offensive`
* `Offensive`

### Language Splits

| Language | Total | Not-Offensive | Offensive |
| :--- | :--- | :--- | :--- |
| Tamil–English | 42,133 | 31,808 | 10,325 |
| Malayalam–English | 18,403 | 17,697 | 706 |
| Kannada–English | 5,874 | 4,397 | 1,477 |

Severe imbalance (especially Malayalam) motivates targeted augmentation.

### Dataset Source

The DravidianCodeMix dataset is publicly available from:

* Website: https://github.com/bharathichezhiyan/DravidianCodeMix-Dataset.git
* Dataset paper: [DravidianCodeMix: sentiment analysis and offensive language identification dataset for Dravidian languages in code-mixed text (Chakravarthi et al., 2022)](https://doi.org/10.1007/s10579-022-09583-7)

Please download the dataset from the official source and place the CSV files locally before running experiments.

---

## 5. Experimental Pipeline

The implementation follows four main stages:

### 1️⃣ Fine-tuning
A pretrained Transformer is fine-tuned on the preprocessed training split.

### 2️⃣ Flip Discovery
For selected examples:
* Extract encoder representation.
* Solve bias homotopy subproblem.
* Perform embedding-space search.
* Decode continuous solution to tokens.
* Validate label flip under original classifier.

### 3️⃣ Augmentation
The flips are appended to the training set.

### 4️⃣ Retraining
The model is retrained on the augmented dataset.

---

## 6. Baseline Comparison

We compare FLAICOL augmentation against:

* No augmentation (baseline)
* SMOTE-based minority oversampling

FLAICOL produces small but consistent improvements in:
* Macro-F1
* Minority-class (Offensive) F1

---

## 7. Results Summary

Across all three language splits:

* Macro-F1 improves modestly (~ +0.5 to +1.0 points)
* Minority-class F1 improves consistently
* XLM-R produces semantically faithful flips
* MuRIL produces more stylistically diverse but noisier flips


---

## 8. Running the Code

Before generating augmented examples, you need to preprocess the raw dataset and train a baseline model. Please follow this sequence of steps carefully:

### Step 1: Dataset Preprocessing
The raw DravidianCodeMix dataset comes with the following granular labels:
* `Not Offensive`
* `Offensive_Untargeted`
* `Offensive_Targeted_Individual`
* `Offensive_Targeted_Group`
* `Offensive_Targeted_Other`
* `Not in indented language`

**Required Actions:**
1. **Filter:** Remove all rows labeled as `Not in indented language`.
2. **Map:** Combine all specific offensive classes (`Offensive_Untargeted`, `Offensive_Targeted_Individual`, `Offensive_Targeted_Group`, `Offensive_Targeted_Other`) into a single `Offensive` class. 
3. Ensure your final preprocessed dataset contains only two labels: `Not Offensive` and `Offensive`.

### Step 2: Initial Fine-Tuning
Using your preprocessed dataset, perform the initial fine-tuning of your chosen Transformer model. 
* Run your standard fine-tuning script on the cleaned CSV.
* **Crucial:** Save the resulting fine-tuned model weights to your local directory. This saved model is mandatory for the flip discovery phase.

### Step 3: Run Flip Generation (`flip.py`)
Once your dataset is cleaned and your baseline model is fine-tuned and saved, you can run the augmentation script. Set your environment variables, ensuring you point to your locally saved model and cleaned dataset:

```bash
# Point MODEL_Path to your saved fine-tuned model path
MODEL_PATH="path/to/your/saved_finetuned_model" \
DATA_CSV="tamil_train_cleaned.csv" \
python flip.py
