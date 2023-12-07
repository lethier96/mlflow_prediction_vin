import warnings
warnings.filterwarnings("ignore")

import os
import configparser

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
from mlflow.models import infer_signature
# import mlflow.sklearn


TARGET = "quality"
RANDOM_STATE = 42

def load_data():

    print("Loading wine dataset")

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    dir_root = os.path.dirname(os.path.abspath(__file__))
    wine_path = os.path.join(dir_root, "data/wine-quality.csv")
    data = pd.read_csv(wine_path)
    print(f"{len(data)} objects loaded")

    return data

def preprocess_data(data):

    print("Preprocessing data")

    # Separate features and target
    X = data.drop(columns=[TARGET])
    y = data[TARGET]

    # Split the data into training and test sets. (0.75, 0.25) split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
    print(f"{len(X_train)} objects in train set, {len(X_test)} objects in test set")

    # do some preprocessing if needed

    return X_train, X_test, y_train, y_test


def train_and_log_model(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Infer an MLflow model signature from the training data (input),
    # model predictions (output) and parameters (for inference).
    signature = infer_signature(X_train, y_train)

    # Log model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="sklearn-model",
        signature=signature)

    # Log model params
    # mlflow.log_params(model.get_params())

    # Log metrics & artifacts to MLflow tracking server
    results = mlflow.evaluate(
        model_info.model_uri,
        data=pd.concat([X_test,y_test], axis=1),
        targets=TARGET,
        model_type="regressor",
        evaluators=["default"]
    )
    return results

def calc_scores_linearmodel(actual, pred):

    rmse = mean_squared_error(actual, pred, squared=False)
    r2 = r2_score(actual, pred)

    return {
        "rmse": rmse,
        "r2": r2}


if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read('config.ini')
    DIR_MLRUNS = config.get('Paths', 'DIR_MLRUNS')
    MODEL_NAME = config.get('Models', 'MODEL_NAME')
    MODEL_VERSION = config.getint('Models', 'MODEL_VERSION')

    mlflow.set_tracking_uri("file:" + os.path.abspath(DIR_MLRUNS))

    np.random.seed(RANDOM_STATE)

    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)

    exp_name = "wine_quality_prediction"
    experiment_id = mlflow.create_experiment(exp_name)
    mlflow.set_experiment(exp_name)

    params_alpha = [0.001, 0.01, 0.1, 1, 10]
    params_l1_ratio = np.arange(0.0, 1.1, 0.2)
    # params_alpha = [0.5]
    # params_l1_ratio = [0.5]

    num_iterations = len(params_alpha) * len(params_l1_ratio)

    run_name = "elastic_net"
    k = 0
    best_score =float('inf')
    best_run_id = None

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id, description=run_name) as parent_run:
        for alpha in params_alpha:
            for l1_ratio in params_l1_ratio:
                k += 1
                print(f"***** ITERATION {k} from {num_iterations} *****")
                child_run_name = f"{run_name}_{k}"
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=RANDOM_STATE)
                with mlflow.start_run(run_name=child_run_name, experiment_id=experiment_id, nested=True) as child_run:
                    results = train_and_log_model(model, X_train, X_test, y_train, y_test)
                    print()
                    mlflow.log_param("alpha", alpha)
                    mlflow.log_param("l1_ratio", l1_ratio)
                    if results.metrics['root_mean_squared_error'] < best_score:
                        best_score = results.metrics['root_mean_squared_error']
                        best_run_id = child_run.info.run_id

    model_uri = f"runs:/{best_run_id}/sklearn-model"
    mv = mlflow.register_model(model_uri, "wine_quality_prediction")
    print("Model saved to registry:")
    print(f"Name: {mv.name}")
    print(f"Version: {mv.version}")
    print(f"Source: {mv.source}")

    print(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}")
    y_pred = model.predict(X_test)
    print(calc_scores_linearmodel(y_test, y_pred))