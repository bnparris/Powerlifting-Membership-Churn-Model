# Powerlifting Churn Prediction

Predicts whether a competitive powerlifter will stop competing the following year, using data from https://openpowerlifting.gitlab.io/opl-csv/files/openipf-latest.zip[OpenPowerlifting]. The model is built for the IPF and affiliated federations, covering sanctioned raw full-power competitions from 2015 onwards.


A gradient boosting classifier is trained on engineered features capturing each lifter's competitive history. The model outputs per-lifter churn probabilities for 2026, as well as whether the 'retention intervention' (a half-price coaching session if the lifter renews their membership) should be offered to the lifter. This is based on the decision framework in `notebooks\07_decision_framework.ipynb`.

---

## Repo Structure

```
├── data/
│   ├── raw/              # OpenPowerlifting source data
│   ├── processed/        # Cleaned and transformed data
│   └── production/       # Panel data and predictions for current season
├── notebooks/            # Modelling decisions and evaluation
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

The target variable is whether a lifter competes in the year following the most recent "complete" calendar year (with a year defined as "complete" from December 7th onwards. The pipeline is designed to be run at the end of the calendar year to make predictions about who will renew their powerlifting membership, as membership operates on a calendar-year basis. Data is aggregated such that each row represents a unique combination of lifter and year. Features are computed based on lifter competition history up until the end of the year specified in the `Year` column. 

**Pre-2015 history**: OpenPowerlifting data prior to 2015 is used for feature lookups only, ensuring time competing and personal best recency are computed correctly for lifters whose careers predate the modelling window.


**Churn definition**: a lifter is considered to have churned if they do not appear in the data the following year. A lifter who takes a year off and returns is still marked as churned for the gap year, reflecting that any year of non-participation is a loss from a retention perspective. A separate reactivation model could be trained to determine lifters likely to return after not competing for one or more calendar years.

**Weight class filtering**: entries are validated against the correct IPF weight class structure for their year. Invalid entries are dropped entirely as they likely reflect data quality issues rather than genuine competition records.


**Age imputation**: age missingness in OpenPowerlifting reflects federation reporting practices rather than lifter behaviour. Missing values are imputed with the training median to prevent the model learning a spurious pattern.

---

## Model Performance

Model performance was primarily evaluated by calculating expected profit from offering retention interventions, after optimising the threshold above which retention interventions should be offered to lifters This can be seen in `notebooks\07_decision_framework.ipynb` and `notebooks\08_test.ipynb`.

---

## How to Run

```bash
pip install -r requirements.txt
python src/train.py
python src/predict.py
```

Requires `config.py`

## Limitations and Future Improvements

Accuracy was used as the evaluation metric for feature selection on the validation set and hyperparameter tuning, assuming equal misclassification costs prior to implementing the decision framework. In retrospect, log loss would have been more appropriate given that the decision framework uses probabilities to determine which lifters to offer retention interventions to. However, the model was found to be well calibrated after calibrating on the validation set despite using accuracy for decision making rather than logloss.

The script necessarily cannot observe true churn labels at prediction time. However, the baseline churn rate above the decision threshold is required to find the necessary size of the control group for the A/B test to have the appropriate statistical power. For this purpose, the baseline churn rate for members above the decision threshold is estimated using their mean churn probability. This approximation relies on the model being well calibrated, as it is shown to be in `notebooks/06_calibration.ipynb`

The decision framework assumed the retention interventions had a constant save rate of $S = 0.15$ (with this assumption being adjusted after the first year of offering interventions using an estimate from the control group). This could be improved upon by using uplift modelling to determine the probability that a retention intervention causes individual churners to stay, rather than using a constant average save rate.

Currently, retraining is manual. Retraining could be automated using GitHub Actions workflow that downloads the latest OpenPowerlifting dataset and automatically trigger the retraining pipeline.
