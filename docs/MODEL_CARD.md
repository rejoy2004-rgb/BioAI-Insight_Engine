# Model card

All numbers below are copied from `data/models/metadata.json`, produced by `python -m src.ml_model` (seed 48, scikit-learn 1.5.0, Python 3.11.9). Re-running with the same seed and library versions reproduces them (checked by a test on synthetic data; on the real data I re-trained twice and got identical test metrics).

## Intended use and non-use

* **Intended:** a teaching/portfolio demonstration of leakage-safe tabular ML, evaluation and explainability.
* **Not intended:** any diagnostic or clinical decision. Not validated on any external population; not a medical device.

## Task and data

Binary classification (benign 0 / malignant 1) from nine 1-10 cytology scores. 699 rows (see [DATA_DICTIONARY.md](DATA_DICTIONARY.md)). **Positive class = malignant**, so *sensitivity* = share of malignant cases found.

## Method

| Step | Choice | Why |
|---|---|---|
| Split | One hold-out fold from `StratifiedGroupKFold(5)` grouped by `Id` -> 558 train / 141 test (53 malignant in test) | keeps class balance; keeps rows from the same sample together |
| Preprocessing | `SimpleImputer(median)` -> `StandardScaler`, **inside each sklearn Pipeline** | fit on training folds only; identical transform at inference; one artifact, so scaler/model mismatch is impossible |
| Models | Majority-class baseline; Logistic Regression; RBF SVM; Random Forest (300 trees) | baseline shows what "no skill" looks like; the others span interpretability vs flexibility |
| Tuning | `GridSearchCV`, 5-fold grouped stratified CV on the **training part only**, refit on ROC-AUC (recall and PR-AUC also logged) | the test set is never used for any decision |
| Primary model | Highest mean CV ROC-AUC, then the **most interpretable model within one standard error** | avoids paying complexity for noise-level gains |
| Operating threshold | Highest threshold <= 0.5 whose **out-of-fold training** recall >= 0.95 -> **0.48** | missing a malignant case is costlier than a false alarm; chosen without test data |
| Final report | Held-out test set, reported once | |

## Results (held-out test set, n = 141; threshold 0.5 unless stated)

| Model | CV ROC-AUC (mean +/- sd, 5 folds) | Test ROC-AUC | PR-AUC | Sensitivity | Specificity | Precision | F1 | Accuracy | Brier |
|---|---|---|---|---|---|---|---|---|---|
| Majority baseline | - | 0.500 | 0.376 | 0.000 | 1.000 | undefined (no positive predictions) | 0.000 | 0.624 | 0.236 |
| **Logistic Regression (primary)** | 0.994 +/- 0.003 | 0.999 | 0.999 | 0.906 (48/53) | 1.000 | 1.000 | 0.950 | 0.965 | 0.024 |
| SVM (RBF) | 0.994 +/- 0.003 | 0.999 | 0.999 | 0.925 (49/53) | 1.000 | 1.000 | 0.961 | 0.972 | 0.021 |
| Random Forest | 0.990 +/- 0.005 | 1.000 | 1.000 | 0.962 (51/53) | 1.000 | 1.000 | 0.981 | 0.986 | 0.015 |

* **Primary model at its operating threshold 0.48:** sensitivity 0.925 (49/53), specificity 1.000 (88/88), precision 1.000, 4 malignant cases missed, 0 false alarms. The 0.95-recall target set on training data did **not** fully transfer to the test set (0.925) - a reminder that with 53 malignant test cases one case is ~1.9 percentage points.
* **Duplicate sensitivity check:** ROC-AUC on the 89 test rows whose feature vector never appears in training: LR 0.998, SVM 0.998, RF 1.000 (baseline 0.5). Duplicates therefore do not explain the high scores.
* **Selection note:** SVM had the top CV ROC-AUC (0.9941 vs 0.9939 for LR); within one standard error, Logistic Regression was chosen as the more interpretable model.
* **Random Forest** has the best point estimates on this test set, but the CV differences between all three models are within noise, and 141 test rows (53 malignant) cannot rank them. I did not pick the model on test performance.
* **Neural network check** (`experiments/dl_model.py`, PyTorch MLP 9-16-1, same split, early stopping on a validation slice of the training data): test ROC-AUC 0.999, sensitivity 0.943, specificity 0.989 (single run, GPU, no repeated seeds). That is indistinguishable from logistic regression, so **no deep model is used in the app**: on 699 rows of 9 ordinal features the extra complexity is unjustified.

### Threshold trade-off (primary model, test set)

| Threshold | Sensitivity | Specificity | Missed malignant | False alarms |
|---|---|---|---|---|
| 0.1 | 100.0 % | 97.7 % | 0 | 2 |
| 0.3 | 94.3 % | 98.9 % | 3 | 1 |
| 0.5 | 90.6 % | 100.0 % | 5 | 0 |
| 0.7 | 84.9 % | 100.0 % | 8 | 0 |

(Full table in the UI and `metadata.json`.)

### Scores are not probabilities

The API field is `malignancy_score`: an **uncalibrated** classifier output. On the test set the primary model's mean Brier score is 0.024 and expected calibration error 0.057, but the reliability bins contain 1-5 samples in the middle bins, which is not enough to claim calibration. No post-hoc calibration (Platt/isotonic) was applied: with 558 training rows the extra layer cannot be validated reliably. The UI therefore never calls the score a probability or "confidence".

## Explainability

SHAP (`TreeExplainer` for Random Forest, `LinearExplainer` for Logistic Regression; units: log-odds for LR). SVM and baseline are reported as "unsupported" instead of approximated. SHAP describes **model behaviour**, not biological causation; with correlated features (cell size and shape are strongly correlated) credit is shared arbitrarily between them.

## Known limitations

* One small, old, single-source dataset; no external validation; no subgroup analysis possible (no demographics).
* Test set is 141 rows; every metric has wide uncertainty (not computed here: no bootstrap confidence intervals were produced).
* Inputs are subjective human scores, so real-world drift in scoring practice is unmodelled.
* Performance on the test set is near ceiling; this says the dataset is easy, not that the model would be safe in practice.

## Reproduce

```bash
python -m src.ml_model --seed 48     # ~1 min on a laptop CPU
python -m experiments.dl_model       # optional, needs torch (requirements-experiments.txt)
```
