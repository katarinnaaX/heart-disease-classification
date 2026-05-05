# Heart Disease Classification
Binary heart disease detection using Bayesian, KNN, and neural network
classifiers on the UCI Heart Disease dataset (920 samples, 10 features).

## Pipeline
1. **Data preprocessing** — handling missing values and invalid zeros; columns with >50% missing data dropped; remaining imputed with median/mode
2. **Feature selection** — information gain + Pearson/Spearman correlation; least informative features excluded
3. **Dimensionality reduction** — LDA used to visualize class separability; classes 1–4 merged into a single "sick" class for better-balanced binary classification
4. **Classification** — minimum-cost Bayesian classifier, KNN (K=11), and neural network (underfitting/overfitting/early stopping/L2 regularization variants compared)

## Results

| Metric      | Bayesian | KNN    | Neural network |
|-------------|:--------:|:------:|:--------------:|
| Accuracy    | 76.76%   | 69.38% | 79.35%         |
| Sensitivity | 83.33%   | 81.18% | 89.22%         |
| Specificity | 68.67%   | 54.74% | 67.07%         |

## Files
- `som.ipynb` — full implementation notebook
- `som.pdf` — detailed project report (Serbian)

## Requirements
tensorflow, numpy, pandas, matplotlib, seaborn, scikit-learn
