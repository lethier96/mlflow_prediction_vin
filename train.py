# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by machine learning from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn


RANDOM_STATE = 42

def load_data():

    print("Loading wine dataset")

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    dir_root = os.path.dirname(os.path.abspath(__file__))
    wine_path = os.path.join(dir_root, "data/wine-quality.csv")
    data = pd.read_csv(wine_path)
    print(f"{len(data)} objects loaded")

    # Separate features and target
    X = data.drop(columns=["quality"])
    y = data["quality"]

    # Split the data into training and test sets. (0.75, 0.25) split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
    print(f"{len(X_train)} objects in train set, {len(X_test)} objects in test set")

    return X_train, X_test, y_train, y_test

def calc_scores(actual, pred):

    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2}


if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    np.random.seed(RANDOM_STATE)

    X_train, X_test, y_train, y_test = load_data()

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(X_train, y_train)
        print(f"Elasticnet model (alpha={alpha:.2f}, l1_ratio={l1_ratio:.2f}):")

        y_pred = lr.predict(X_test)
        scores = calc_scores(y_test, y_pred)

        for metric, value in scores.items():
            mlflow.log_metric(metric, value)
            print(f"{metric}: {value:.2f}")

        mlflow.sklearn.log_model(lr, "model")
