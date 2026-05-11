# Predicting Powerlifter Churn

## Motivation

HistGradientBoostingClassifier trained to identify powerlifters at risk of churning (i.e. not competing the following calendar year). (Mirrors powerlifting federation membership which spans a calendar year.) This enables powerlifting federations to employ targeted retention interventions for lifters likely to churn. Classifier built for IPF (International Powerlifting Federation) and its affiliate federations, covering sanctioned, raw, full-power meets from 2015 onwards. Classifier was trained on data from Open Powerlifting. achieved an accuracy of X%, compared to a majority class baseline of Y%. In practice, if business data were available, a profit-based evaluation metric taking into account customer liftertime value, cost of intervention retentions, and effectiveness of interventions could be used to better align model performance with real-world impact.

## Overview and key decisions

-  The data is aggregated to one row per lifter per year.
-  Features are constructed using lifters' competition history up to and including the end of the `Year` column.
-  The target variable is whether a lifter competes in the year following that specified in the `Year` columns.
## Dataset
The original dataset was transformed to a panel structure as detailed below.

Source| https://openpowerlifting.gitlab.io/opl-csv/bulk-csv.html
Original Structure| A history of powerlifting meet performances. Each record corresponds to a lifter's performance at a particular powerlifting meet.
Transformed panel structure| Panel data where each record represents a unique combination of lifter and year. Features are constructed using only information available up to the end of that year. Contains X records and Y unique lifters. 



### Cleaning

See `notebooks/01_cleaning.ipynb` for full implementation.

A reduced set of 26 columns was retained. Routine cleaning was applied including type standardisation and whitespace normalisation. Entries sharing the same name, date, sex, total and event and meet name were treated as duplicate records. Duplicates were removed on this subset of columns to account for minor inconsistencies in other fields. GIVEN PLS ONLY COMPETE OCCASIONALLY.
#### Age imputation
Missing `Age` (~25% of dataset) imputed with median: Missing age is not expected to consistently predict churn and is likely driven by data entry practices, which may change over time. Allowing the model to learn from missingness directly would risk capturing patterns that do not generalize. While missingness in 'Age' may correlate with federation, this information is more directly and stably captured through the `FederationProportion` feature. Note `FederationProportion` was not retained during feature selection, suggesting that federation-related effects were not a significant driver in the final model.

Rows with missing WeightClassKg were removed (1.17% of the original dataset). Given the small proportion of missing data, row-wise deletion was unlikely to introduce meaningful bias. Historical IPF weight classes from 2015 onwards were inferred from the dataset (See `notebooks\01_cleaning.ipynb` for implementation) and entries with `WeightClassKg` that did not match historical classes were removed (X% of original dataset). `WeightClassKg` not matching historical IPF weight classes could indicate a competition from a non-IPF federation EXPAND. This also makes the cleaned dataset more suitable as a general-purpose resource for other analyses of IPF competition history.

Entries exceeding current official IPF world records by weight class were 
flagged and reviewed. These were retained after cross-referencing against 
available meet results, as records can only be set at international 
competitions and higher totals may occur at national or local level without 
official recognition.

### Filtering
- **Equipment**: restricted to raw powerlifting only, as equipped lifting is 
  considered a distinct discipline
- **Sanctioned**: unsanctioned meets excluded
- **Division** (e.g. Sub-Juniot, Junior, Open, Masters): dropped due to high cardinality (1,135 unique values) caused by inconsistent data entry. This signal is also captured by the `Age` feature.
- **Event**: restricted to full power (SBD) entries, excluding single-lift 
  competitions such as bench-only




## Feature Engineering

See `notebooks/02_data_transformation.ipynb` for full implementation.

Predictions are made at the end of each calendar year, as most powerlifting 
federation memberships operate on an annual basis. All features are constructed 
using only information available up to the end of that year to prevent leakage.

### Final Feature Set

| Feature | Description |
|---|---|
| `TimeSinceLastPBYearEnd` | Days since the lifter last set a personal best, measured at year end |
| `BestGoodliftOfYear` | Best Goodlift points achieved across all meets that year |
| `ImprovementGradientWithinYear` | Rate of improvement in total between first and last meet of the year |
| `Age` | Lifter age at time of last meet of the year |
| `AvgMeetsPerYear` | Expanding mean of meets per year up to and including the current year |
| `Sex` | Encoded as 0 (F), 1 (M), 2 (Mx) |

### Feature Construction Notes
- **Improvement features**: both absolute and percentage-based versions were 
  constructed; absolute versions were retained after validation performance 
  showed negligible difference between the two
- **Attempt imputation**: where individual attempt data was absent but the 
  lifter did not bomb out, `AttemptsMade` was imputed using the mean of 
  non-bombing lifters
- **Age imputation**: a binary `AgeMissing` indicator was added before imputing 
  missing `Age` values with the median for that year, to allow the model to 
  learn from the missingness pattern while avoiding leakage
- **Weight class and federation** were considered but did not survive feature 
  selection

### Target Variable
`Churns` is set to 1 where a lifter does not appear in the dataset in the 
following calendar year. The final year (2025) is excluded as future 
competition status is unknown.

## Methodology

### Train / Validation / Test Split
A temporal split was used to prevent data leakage:

| Split      | Years     | Share |
|---|---|---|
| Train      | up to 2022 | ~75% |
| Validation | 2023       | ~12% |
| Test       | 2024       | ~13% |

Feature selection decisions were made on the validation set. For hyperparameter 
tuning, train and validation years were combined with time-based cross-validation 
applied within the combined set. Final evaluation was performed on the held-out 
2024 test set.

### Handling Covid Years
Excluding 2020 and 2021 from training was considered given atypical churn 
patterns during the pandemic. Performance differences on the validation set 
were negligible between models trained with and without these years, so they 
were retained. This is expected to make the model more robust to future shocks.

### Model Selection
`HistGradientBoostingClassifier` was selected as it natively handles 
categorical features and missing values.

### Evaluation Metric
Accuracy was chosen as the primary evaluation metric. The test set is 
approximately balanced (52:48), making accuracy an interpretable metric. 
In the absence of business data (intervention cost, conversion rate, customer 
lifetime value), equal misclassification costs were assumed. A profit-based 
metric could be substituted with this information.

### Feature Selection
1. Permutation importance was computed on the validation set and features 
   with negative importance were removed
2. Remaining features were added incrementally in order of importance and 
   validation accuracy was tracked
3. The feature count maximising validation accuracy was selected (6 features)

## Results
Key metrics and findings.

## How to Run
Steps to reproduce.

## Project Structure
Folder layout.

## Requirements
Dependencies.

## Future Work
Planned improvements.

## Author
Your name + links.