"""
Purpose: Functions for fairness analyses.
Author: Mélina Verger
Reviewed by: Sébastien Lallé

Configuration on which the code was tested: Python 3.10.4 on macOS 14.1 (with 16 GB of RAM).
"""

import numpy as np
import pandas as pd
import pickle

# Fairness metrics
from sklearn.metrics import roc_auc_score, f1_score
from maddlib import evaluation


MODEL = "RandomForestClassifier"


# ========== UTILS FUNCTIONS ==========


# Match the predictions with their related region IDs
def get_y_pred_region(X_test, y_pred, id_region_file="data_regions.csv", path="./"):
    """
    Given some predictions, return them with their associated region IDs (and student IDs if needed).
    """
    data_regions = pd.read_csv(path + id_region_file)[["student_id", "region_group"]]
    regions_in_pred = X_test.merge(data_regions, how='inner', on="student_id", copy=True)[["student_id", "region_group"]]
    y_pred_region = pd.concat([y_pred.reset_index(), regions_in_pred], axis=1)
    return y_pred_region


# Separate the predictions according to all groups, 2 groups, or 1 group vs. the rest
def separate_y_pred_region(y_pred_region, groups="all"):
    """
    Given some tagged predictions, return sets of predictions according to some groups.

    Examples:
        groups = ("NCR", "LUZON")
        groups = "NCR" # means vs. the rest
        groups = "all"
    """

    regions = ["NCR", "LUZON", "MINDANAO", "VISAYAS", "ABROAD"]

    y_pred_groups = list()

    if type(groups) == str:
        if groups == "all":
            for g in regions:
                y_pred_groups.append(y_pred_region[y_pred_region["region_group"] == g])
        elif groups in regions:  # groups vs. the rest
            y_pred_groups.append(y_pred_region[y_pred_region["region_group"] == groups])
            y_pred_groups.append(y_pred_region[y_pred_region["region_group"] != groups])
        else:
            raise Exception("`groups` argument as a str could be 'all' or a valid region name.")

    elif type(groups) == tuple:
        group0, group1 = groups
        if (group0 in regions) and (group1 in regions): 
            y_pred_groups.append(y_pred_region[y_pred_region["region_group"] == group0])
            y_pred_groups.append(y_pred_region[y_pred_region["region_group"] == group1])
        else:
            raise Exception("One of the group names of `groups` argument is not valid.")

    else:
        raise Exception("`groups` argument is not valid.")
    
    return y_pred_groups


# ========== EVALUATION FUNCTIONS ==========

def fairness_evaluation(model_name):
    # Load model
    model = pickle.load(open("./models/" + model_name, "rb"))

    # Load validation subsets
    X_val_sets = list()
    y_val_sets = list()

    n_splits = 10
    for i_split in range(0, n_splits):
        X_val = pickle.load(open("./folds/"+ "X_validate_" + "fold" + str(i_split), "rb"))
        y_val = pickle.load(open("./folds/"+ "y_validate_" + "fold" + str(i_split), "rb"))

        X_val_sets.append(X_val)
        y_val_sets.append(y_val)

    # Compute the results
    regions = ["NCR", "LUZON", "MINDANAO", "VISAYAS", "ABROAD"]

    auc = dict()
    f1 = dict()
    madd = dict()

    for num_group in range(len(regions)):
        auc[regions[num_group]] = list()
        f1[regions[num_group]] = list()
        madd[regions[num_group]] = list()

    for i_split in range(0, n_splits):

        # A. Separate the predictions according to the region groups

        # Select one validation set
        X_val = X_val_sets[i_split]
        y_val = y_val_sets[i_split]

        # Get the predictions
        y_pred = model.predict(X_val)

        # Separate the ground truth and the predictions according to the region groups
        y_pred_tagged = get_y_pred_region(X_val, pd.DataFrame(y_pred))
        y_val_tagged = get_y_pred_region(X_val, pd.DataFrame(y_val))
        
        y_pred_groups = separate_y_pred_region(y_pred_tagged, "all")
        y_val_groups = separate_y_pred_region(y_val_tagged, "all")

        # B. Compute fairness metrics on the prediction sets

        for num_group in range(len(regions)):
            # AUC
            auc[regions[num_group]].append(roc_auc_score(y_val_groups[num_group]["grade"], y_pred_groups[num_group].iloc[:, 1]))

            # F1
            f1[regions[num_group]].append(f1_score(y_val_groups[num_group]["grade"], y_pred_groups[num_group].iloc[:, 1], 
                                        average='weighted'))

            # MADD
            y_pred_g0, y_pred_g1 = separate_y_pred_region(y_pred_tagged, regions[num_group])
            madd[regions[num_group]].append(evaluation.MADD(h="auto", 
                                        pred_proba_sf0=np.array(y_pred_g0.iloc[:, 1]), 
                                        pred_proba_sf1=np.array(y_pred_g1.iloc[:, 1])))
    
    # Print the results
    print("===== Number of samples and students =====")
    for num_group in range(len(regions)):
        print(y_val_groups[num_group]["region_group"].value_counts())
        print("Students:", len(y_val_groups[num_group]["student_id"].unique()))
        print()

    print("===== AUC (average and std) =====")
    for num_group in range(len(regions)):
        print(regions[num_group])
        print(np.mean(auc[regions[num_group]]))
        print(np.std(auc[regions[num_group]]))
        print()
    
    print("===== F1-score (average and std) =====")
    for num_group in range(len(regions)):
        print(regions[num_group])
        print(np.mean(f1[regions[num_group]]))
        print(np.std(f1[regions[num_group]]))
        print()
    
    print("===== MADD (average and std) =====")
    for num_group in range(len(regions)):
        print(regions[num_group])
        print(np.mean(madd[regions[num_group]]))
        print(np.std(madd[regions[num_group]]))
        print()

if __name__ == '__main__':
    fairness_evaluation(MODEL)
