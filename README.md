# Powerlifting Churn Prediction

Predicts whether a competitive powerlifter will stop competing the following year, using data from https://openpowerlifting.gitlab.io/opl-csv/files/openipf-latest.zip[OpenPowerlifting]. Built for the IPF and affiliated federations, covering sanctioned raw full-power competitions from 2015 onwards.

A gradient boosting classifier is trained on engineered features capturing each lifter's competitive history. The model outputs per-lifter churn probabilities for 2026.

---

## Repo Structure

```
├── data/
│   ├── raw/              # OpenPowerlifting source data
│   ├── processed/        # Cleaned and transformed data
│   └── production/       # Panel data and predictions for current season
├── notebooks/            # Exploratory analysis and modelling decisions
├── src/
│   ├── cleaning.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
└── models/               # Saved model and metadata
```

The `.py` files contain the production pipeline. The notebooks document the decisions behind the pipeline (cleaning choices, feature engineering, feature selection, and hyperparameter tuning).

---

## Overview & Key Decisions

The target variable is whether a lifter competes in the year following the most recent "complete" calendar year (with a year defined as "complete" from December 7th onwards. This can be changed in XXX). The pipeline is designed to be run at the end of the calendar year to make predictions about who will renew their powerlifting membership, as membership operates on a calendar-year basis. Data is aggregated such that each row represents a unique combination of lifter and year. Features are computed based on lifter competition history up until the end of the year specified in the `Year` column. 

**Pre-2015 history**: OpenPowerlifting data prior to 2015 is used for feature lookups only, ensuring time competing and personal best recency are computed correctly for lifters whose careers predate the modelling window.


**Churn definition**: a lifter is considered to have churned if they do not appear in the data the following year. A lifter who takes a year off and returns is still marked as churned for the gap year, reflecting that any year of non-participation is a loss from a retention perspective. A separate reactivation model could be trained to determine lifters likely to return after not competing for one or more calendar years.

**Weight class filtering**: entries are validated against the correct IPF weight class structure for their year. Invalid entries are dropped entirely as they likely reflect data quality issues rather than genuine competition records.


**Age imputation**: age missingness in OpenPowerlifting reflects federation reporting practices rather than lifter behaviour. Missing values are imputed with the training median to prevent the model learning a spurious pattern.

---

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | |
| ROC-AUC | |
| F1 | |
| Precision | |
| Recall | |

*Evaluated on held-out test set (2024 season).*

---

## How to Run

```bash
pip install -r requirements.txt
python src/train.py
python src/predict.py
```

Requires `config.py` — see `config.example.py`.

## Limitations and Future Improvements

Accuracy was used as the evaluation metric for feature selection on the validation set and hyperparameter tuning, assuming equal misclassification costs. In retrospect ROC-AUC would have been more appropriate given that the optimal decision threshold for retention interventions depends on the probabilities outputted by the model. Using ROC-AUC would have meant that a decision threshold was not assumed in the training process.

Year column included in hyperparameter tuning to allow for cross validation splits to be done using Year column but not used as a feature in the model. Permutation importance on Year column was found to be 0 so its inclusion is unlikely to AFFECT?EFFECT? model accuracy and therefore unlikely to influence selection of hyperparameters. Year was not included as a feature in the final model as it would add noise due to permutation importance being 0. 




Currently, retraining is manual. Retraining could be automated using GitHub Actions workflow that downloads the latest OpenPowerlifting dataset. 