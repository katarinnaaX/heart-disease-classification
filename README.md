# Heart Disease Classification – Decision Support System

Bayesian, KNN, and neural network classifiers for binary heart disease detection
on the UCI Heart Disease dataset (920 samples, 13 features).

## Approach

The pipeline covers the full machine-learning workflow from raw data to comparative
evaluation:

**Data preprocessing** — invalid zero values in `trestbps`, `chol`, and `thalach`
were replaced with NaN, as were all `?` entries. Columns `ca` and `thal` were
dropped entirely (>50% missing). Remaining missing values were imputed with the
median (continuous features) or mode (discrete features).

**Feature selection** — Information gain (IG) was computed for all 11 features.
`fbs` ranked last (IG = 0.013) and was excluded. Correlation analysis using both
Pearson and Spearman methods confirmed that `cp` (chest pain type) and `oldpeak`
(ST depression) carry the strongest signal.

**Dimensionality reduction (LDA)** — 5-class projection showed the healthy class
(0) is reasonably separable, but classes 1–4 overlap heavily. To address class
imbalance and poor separability, all disease classes were merged into a single
"sick" class, reducing the problem to binary classification.

**Classifiers evaluated:**

- *Minimum-cost Bayesian classifier* — assumes Gaussian features; uses a
  cost-asymmetric decision threshold (c₁₂ = 10×c₂₁) to heavily penalize
  missed detections of sick patients.
- *K-Nearest Neighbors (KNN)* — non-parametric; optimal K found by sweeping
  K = 1…200 on a stratified 67/33 split.
- *Neural network (Keras)* — 10-input, 1-output sigmoid network; three
  structural variants compared (underfitting / well-fitted / overfitting) plus
  early stopping and L2 regularisation to control overfitting.

## Results

| Metric        | Bayesian (min-cost) | KNN (K=11) | Neural network |
|---------------|:-------------------:|:----------:|:--------------:|
| Accuracy      | 76.76%              | 69.38%     | **79.35%**     |
| Precision     | 76.58%              | 69.00%     | 77.12%         |
| Sensitivity   | 83.33%              | 81.18%     | **89.22%**     |
| Specificity   | **68.67%**          | 54.74%     | 67.07%         |
| False-negative rate | 16.67%        | 18.82%     | **10.78%**     |

The neural network achieved the highest accuracy and sensitivity (fewest missed
sick patients), making it the best fit for clinical screening where false negatives
carry the greatest cost. The Bayesian classifier offered the most balanced and
stable performance; KNN was the weakest due to the small, imbalanced dataset.

## Files

- `som.ipynb` — full implementation notebook (Serbian)
- `som.pdf` — detailed project report (Serbian)

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
keras / tensorflow
```
