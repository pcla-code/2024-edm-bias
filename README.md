# Overview

This repository contains supplementary materials for the following conference paper:

Anonymous Authors.\
Evaluating Algorithmic Bias in Models for Predicting Academic Performance of Filipino Students.\
Submitted for review in the 17th International Conference on Educational Data Mining ([EDM 2024](https://educationaldatamining.org/edm2024/)).

# File structure

## `step1_get_student_ids.py`

Select and normalize Canvas student IDs that will be used later.

(Data not included due to not obtaining approval yet, see Section 5 in the paper.)

## `step2_create_feature_data_set.py`

Compute the values of predictor and target variables from the raw Canvas data.\
See Section 3.3 in the paper.

(Data not included due to not obtaining approval yet, see Section 5 in the paper.)

## `step3_prediction_modeling.py`

Perform prediction modeling.\
See Section 3.4 in the paper.

```$ python3 step3_prediction_modeling.py -sc <True|False> -c <True|False> -o <True|False>```

## `step4_fairness_analysis.py`

Perform fairness analysis (summarized in `results_grid.xlsx`).\
See Section 3.5 in the paper.
