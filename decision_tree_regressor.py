import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error

import mlflow
from mlflow.tracking import MlflowClient

import common as common

DATA_PATH = common.CONFIG['paths']['data']
DIR_MLRUNS = common.CONFIG['paths']['mlruns']

TARGET = common.CONFIG['ml']['target_name']
RANDOM_STATE = common.CONFIG['ml']['random_state']

EXPERIMENT_NAME = common.CONFIG['mlflow']['experiment_name']
MODEL_NAME = common.CONFIG['mlflow']['model_name']
MODEL_VERSION = common.CONFIG['mlflow']['model_version']

def load_data():
    print("Loading wine dataset")
    data = pd.read_csv(DATA_PATH)
    print(f"{len(data)} objects loaded")
    return data

def preprocess_data(data):
    print("Preprocessing data")
    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
    print(f"{len(X_train)} objects in train set, {len(X_test)} objects in test set")
    return X_train, X_test, y_train, y_test

def train_and_log_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    signature = mlflow.models.infer_signature(X_train, y_train)
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="sklearn-model",
        signature=signature)
    results = mlflow.evaluate(
        model_info.model_uri,
        data=pd.concat([X_test, y_test], axis=1),
        targets=TARGET,
        model_type="regressor",
        evaluators=["default"]
    )
    return results

def get_registered_model_mse(model_name, model_version):
    client = MlflowClient()
    model_version_details = client.get_model_version(name=model_name, version=model_version)
    run_id = model_version_details.run_id
    run_data = client.get_run(run_id).data
    return run_data.metrics['root_mean_squared_error']

if __name__ == "__main__":
    mlflow.set_tracking_uri("file:" + DIR_MLRUNS)
    np.random.seed(RANDOM_STATE)
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)
    exp_name = "wine_quality_prediction"
    mlflow.set_experiment(exp_name)

    params_max_depth = [None, 10, 20]
    params_min_samples_split = [2, 5]
    num_iterations = len(params_max_depth) * len(params_min_samples_split)

    run_name = "decision_tree_regressor"
    k = 0
    best_score = float('inf')
    best_run_id = None

    with mlflow.start_run(run_name=run_name, description=run_name) as parent_run:
        for max_depth in params_max_depth:
            for min_samples_split in params_min_samples_split:
                k += 1
                print(f"\n***** ITERATION {k} from {num_iterations} *****")
                child_run_name = f"{run_name}_{k:02}"
                model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=RANDOM_STATE)
                with mlflow.start_run(run_name=child_run_name, nested=True) as child_run:
                    results = train_and_log_model(model, X_train, X_test, y_train, y_test)
                    mlflow.log_param("max_depth", max_depth)
                    mlflow.log_param("min_samples_split", min_samples_split)
                    if results.metrics['root_mean_squared_error'] < best_score:
                        best_score = results.metrics['root_mean_squared_error']
                        best_run_id = child_run.info.run_id
                    print(f"rmse: {results.metrics['root_mean_squared_error']}")
                    print(f"r2: {results.metrics['r2_score']}")


    # Compare with the registered model
    registered_model_rmse = get_registered_model_mse(MODEL_NAME, MODEL_VERSION)
    print(f"Registered model RMSE: {registered_model_rmse}")
    print(f"Best model RMSE: {best_score}")

    if best_score < registered_model_rmse:
        model_uri = f"runs:/{best_run_id}/sklearn-model"
        mv = mlflow.register_model(model_uri, MODEL_NAME)
        print("Model saved to the model registry:")
        print(f"Name: {mv.name}")
        print(f"Version: {mv.version}")
        print(f"Source: {mv.source}")
    else:
        print("The new model did not improve the MSE and was not registered.")

    print("#" * 20)
    print("Load model from the model registry")
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    print(f"Model URI: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri=model_uri)
    y_pred = model.predict(X_test)
    print(f"RMSE for test data = {root_mean_squared_error(y_test, y_pred, squared=False)}")