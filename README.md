# Overview

This repository contains supplementary materials for the following conference paper:

Valdemar Švábenský, Mélina Verger, Maria Mercedes T. Rodrigo, Clarence James G. Monterozo, Ryan S. Baker, Miguel Zenon Nicanor Lerias Saavedra, Sébastien Lallé, and Atsushi Shimada.\
**Evaluating Algorithmic Bias in Models for Predicting Academic Performance of Filipino Students**.\
In Proceedings of the 17th International Conference on Educational Data Mining ([EDM 2024](https://educationaldatamining.org/edm2024/)).

# File structure

### `step1_get_student_ids.py`

Select and normalize Canvas student IDs that will be used later.

(We are unable to publish the input data due to the regulations of Ateneo de Manila University.)

### `step2_create_feature_data_set.py`

Compute the values of predictor and target variables from the raw Canvas data.\
See Section 3.3 in the paper.

(We are unable to publish the input data due to the regulations of Ateneo de Manila University.)

### `step3_prediction_modeling.py`

Perform prediction modeling.\
See Section 3.4 in the paper.

```$ python3 step3_prediction_modeling.py -sc <True|False> -c <True|False> -o <True|False>```

### `step4_fairness_analysis.py`

Perform fairness analysis.\
See Section 3.5 in the paper.

# How to cite

If you use or build upon the materials, please use the BibTeX entry below to cite the original paper (not only this web link).

```bibtex
@inproceedings{Svabensky2024evaluating,
    author    = {\v{S}v\'{a}bensk\'{y}, Valdemar and Verger, M\'{e}lina and Rodrigo, Maria Mercedes T. and Monterozo, Clarence James G. and Baker, Ryan S. and Saavedra, Miguel Zenon Nicanor Lerias and Lall\'{e}, S\'{e}bastien and Shimada, Atsushi},
    title     = {{Evaluating Algorithmic Bias in Models for Predicting Academic Performance of Filipino Students}},
    booktitle = {Proceedings of the 17th International Conference on Educational Data Mining},
    series    = {EDM '24},
    location  = {Atlanta, GA, USA},
    publisher = {International Educational Data Mining Society},
    month     = {07},
    year      = {2024},
    numpages  = {8},
}
```
