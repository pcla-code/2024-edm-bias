"""
Purpose: Use the data set from the previous step to build regression/classification models.
Authors: Valdemar Švábenský (core), Mélina Verger (command-line execution), Sébastien Lallé (hyperparameter search)
Reviewed by: Clarence James G. Monterozo

Configuration on which the code was tested: Python 3.10 on Windows 11 (with 8 GB of RAM).
"""

import pandas as pd
import numpy as np
import pickle
import argparse

# Regression, classification algorithms
from xgboost import XGBRegressor, XGBClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.dummy import DummyRegressor, DummyClassifier

# Cross-validation and tuning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import GridSearchCV

# Performance evaluation
from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score
from scipy.stats import spearmanr
from statistics import mean, stdev

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def prepare_data(filename, feature_scaling=False):
    """
    Read the input CSV file with the predictor and target variables and prepare them for further analysis.
    :param: filename (str). Path to the input file.
    :param: feature_scaling (bool). Whether to apply feature scaling.
    :return: (DataFrame, DataFrame). X vector of features and y vector of labels.
    """
    X = pd.read_csv(filename)
    y = X.pop('grade')
    if BINARY_CLASSIFICATION:
        y = y.apply(lambda grade: 1 if grade < 0.721 else 0)

    # X.drop(columns=[
    #     'avg_quiz_submitted_gap'
    #     ], inplace=True)

    if feature_scaling:
        columns_to_scale = X.columns.drop('student_id', 'course_id')
        X[columns_to_scale] = StandardScaler().fit_transform(X[columns_to_scale])

    # X.info()
    return X, y


# ===== HELPER FUNCTIONS FOR MODEL TRAINING AND CROSS-VALIDATION =====

def create_student_cv_groups(X):
    """
    Code by Miggy Andres-Bray (https://www.miggyandresbray.com/) to be able to achieve
    student-level cross-validation during model training.

    We create a NumPy array `groups` of size equal to the number of rows in the input dataset `X`,
    such that each element of `groups` is the index of the first occurrence of a student's ID at that place in `X`.

    Example: Assume these student IDs in the input dataset: [A, B, A, C, B]
    Expected output (result of `groups`): [0. 1. 0. 2. 1]

    :param: X (DataFrame). Dataframe of features that also include student ID.
    :return: (np.array). Array of IDs showing which data points belong to the same students.
    """
    group_dict = {}
    groups = np.array([])
    for index, row in X.iterrows():
        student_id = int(row['student_id'])
        if student_id not in group_dict:
            group_dict[student_id] = index
        groups = np.append(groups, group_dict[student_id])
    return groups


def test_model_performance(X_test, y_test, model):
    """
    Have the model predict the test set and evaluate the model performance.

    :param: X_test. Predictor variable values for the test set.
    :param: y_test. Actual target variable values for the test set.
    :return: Model performance metrics on the test set.
    """
    y_test_predicted = model.predict(X_test)
    if not BINARY_CLASSIFICATION:
        rmse = mean_squared_error(y_test, y_test_predicted, squared=False)
        spearman = spearmanr(y_test, y_test_predicted).correlation
        return rmse, spearman
    else:
        auc = roc_auc_score(y_test, y_test_predicted)
        f1 = f1_score(y_test, y_test_predicted, average='weighted')
        return auc, f1


def train_and_cross_validate_model(X, y, groups, model, grid=None, num_folds=10, num_nested_folds=5):
    """
    Using the functions above, train and cross-validate the given model.

    :param: X. Predictor variable values.
    :param: y. Target variable values.
    :param: groups. Results of create_student_cv_groups(X).
    :param: model. The model object.
    :param: grid. The object for hyperparameter grid search.
    :param: num_folds. The number of folds for the outer cross-validation.
    :param: num_nested_folds. The number of folds for the inner cross-validation.
    :return: The average of model performance metrics across the <num_folds> cross-validation.
    """
    metric1_values, metric2_values = [], []  # Individual (= per each fold) performance metrics values
    gkf = StratifiedGroupKFold(n_splits=num_folds)
    i_split = 0
    for train_index, validate_index in gkf.split(X, y, groups=groups):
        # Create the data subsets (folds) and optionally export them to a file saved locally
        X_train, X_validate = X.loc[train_index], X.loc[validate_index]
        y_train, y_validate = y[train_index], y[validate_index]
        if OUTPUT_FILES: 
            pickle.dump(X_train, open("./folds/" + "X_train_" + "fold" + str(i_split), "wb"))
            pickle.dump(y_train, open("./folds/" + "y_train_" + "fold" + str(i_split), "wb"))
            pickle.dump(X_validate, open("./folds/" + "X_validate_" + "fold" + str(i_split), "wb"))
            pickle.dump(y_validate, open("./folds/" + "y_validate_" + "fold" + str(i_split), "wb"))

        # Either fit the given model as-is, or search for hyperparameters within the training set using nested cross-val
        if not grid:  # Use the provided values of hyperparameters
            model.fit(X_train, y_train)  # Sk-learn model objects are always re-fitted freshly in-place
            metric1, metric2 = test_model_performance(X_validate, y_validate, model)
        else:  # Perform hyperparameter tuning
            groups_train = groups[train_index]
            gridmodel = GridSearchCV( 
                estimator=model,
                param_grid=grid,
                cv=StratifiedGroupKFold(n_splits=num_nested_folds, shuffle=True).split(X_train, y_train, groups_train),
                scoring='roc_auc',
                n_jobs=-1).fit(X_train, y_train)
            metric1, metric2 = test_model_performance(X_validate, y_validate, gridmodel.best_estimator_)

        metric1_values.append(metric1)
        metric2_values.append(metric2)
        i_split += 1

    # If the line below is commented out, the model stays trained on the last fold
    # model.fit(X, y)  # Fit the final Sk-learn model on the whole training set for the test set
    return metric1_values, metric2_values


# ===== MAIN =====

def main_output_results(model, metric1_values, metric2_values):
    model_name = model.__class__.__name__
    metric1_label = 'RMSE' if not BINARY_CLASSIFICATION else 'AUC'
    metric2_label = 'Spearman' if not BINARY_CLASSIFICATION else 'F1'

    # Print the cross-validation results
    print('***********', model_name, '***********')
    print(metric1_label + '_avg', mean(metric1_values))
    print(metric1_label + '_stdev', stdev(metric1_values))
    if model_name != 'DummyRegressor':
        # Can't compute Spearman for the dummy model since it has 0 variance in the predictions
        print(metric2_label + '_avg', mean(metric2_values))
        print(metric2_label + '_stdev', stdev(metric2_values))
    print()

    # Save the trained models
    if OUTPUT_FILES:
        pickle.dump(model, open("./models/" + str(model_name), "wb"))


def main(feature_scaling):
    X, y = prepare_data('step2-feature-computation/data_preprocessed.csv', feature_scaling)  # Training set
    groups = create_student_cv_groups(X)

    if not BINARY_CLASSIFICATION:
        dummy = DummyRegressor(strategy='mean')
        modelDT = DecisionTreeRegressor(max_depth=18)
        modelRF = RandomForestRegressor(max_depth=20, max_features=None, n_estimators=10)
        modelKNN = KNeighborsRegressor(n_neighbors=20, weights='distance', p=1)
        modelXGB = XGBRegressor(scale_pos_weight=0.92, max_delta_step=0.73, max_depth=11)
        modelLIN = Ridge()
    else:
        gridspaceDT = {
                'max_depth': [i for i in range(1, 21, 2)]
            }
        gridspaceRF = {
                'max_depth': [i for i in range(1, 21, 2)],
                'n_estimators': [i for i in range(10, 101, 10)]
            }
        gridspaceKNN = {
                'n_neighbors': [i for i in range(1, 21, 2)]
            }
        gridspaceXGB = {
                'max_depth': [i for i in range(1, 21, 2)],
                'scale_pos_weight': [i/10.0 for i in range(1, 10)]
            }
        gridspaceLIN = {
                'C': [100.0, 50.0, 25.0, 10.0, 5.0, 1.0, 0.5, 0.1]
            }

        # These hyperparameters are used only when the grid search is not enabled (i.e., grid=None)
        dummy = DummyClassifier()
        modelDT = DecisionTreeClassifier(criterion='log_loss', max_depth=21)
        modelRF = RandomForestClassifier(criterion='log_loss', max_features=None, n_estimators=10)
        modelKNN = KNeighborsClassifier(n_neighbors=8, p=1)
        modelXGB = XGBClassifier(scale_pos_weight=0.62, max_delta_step=1.05, max_depth=13, colsample_bytree=0.95)
        modelLIN = LogisticRegression(solver='liblinear', penalty='l1', class_weight='balanced')

    models = [modelDT, modelRF, modelKNN, modelXGB, modelLIN]  # dummy, modelDT, modelRF, modelKNN, modelXGB, modelLIN
    grids = [gridspaceDT, gridspaceRF, gridspaceKNN, gridspaceXGB, gridspaceLIN]

    # for model in models:
    for model, grid in zip(models, grids):
        metric1_values, metric2_values = train_and_cross_validate_model(X, y, groups, model, grid)
        main_output_results(model, metric1_values, metric2_values)


# ===== ADDITIONAL SUPPORT FOR COMMAND-LINE EXECUTION =====

def process_binary_arg(arg):
    if arg is True or arg == "True":
        return True
    if arg is False or arg == "False":
        return False
    raise ValueError("The argument", arg, "should be True or False.")


def command_line_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--classification", help="Perform binary classification.", default=True)
    parser.add_argument("-sc", "--scaling", help="Perform feature scaling.", default=True)
    parser.add_argument("-o", "--output_saved", help="Save output to file.", default=False)

    args = parser.parse_args()
    bool_classification = process_binary_arg(args.classification)
    bool_scaling = process_binary_arg(args.scaling)
    bool_output = process_binary_arg(args.output_saved)
    return bool_classification, bool_scaling, bool_output


if __name__ == '__main__':
    BINARY_CLASSIFICATION, FEATURE_SCALING, OUTPUT_FILES = command_line_options()
    main(FEATURE_SCALING)
