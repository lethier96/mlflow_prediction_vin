import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

import mlflow

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
    # y_pred = model.predict(X_test)

    # Infer an MLflow model signature from the training data (input),
    # model predictions (output) and parameters (for inference).
    signature = mlflow.models.infer_signature(X_train, y_train)

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

if __name__ == "__main__":

    # Using MLflow Tracking locally: everything will be stored in DIR_MLRUNS folder
    mlflow.set_tracking_uri("file:" + DIR_MLRUNS)

    # for reproducibility reasons
    np.random.seed(RANDOM_STATE)

    # load data from csv file
    data = load_data()
    # preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # set mlflow experiment
    exp_name = "wine_quality_prediction"
    # experiment_id = mlflow.create_experiment(exp_name)
    mlflow.set_experiment(exp_name)

    params_alpha = [0.01, 0.1, 1, 10]
    params_l1_ratio = np.arange(0.0, 1.1, 0.5)
    # params_alpha = [0.5]
    # params_l1_ratio = [0.5]

    num_iterations = len(params_alpha) * len(params_l1_ratio)

    run_name = "elasticnet"
    k = 0
    best_score =float('inf')
    best_run_id = None

    # Test all the defined combinations of hyperparams
    # Log each run
    # Register the best model
    with mlflow.start_run(run_name=run_name, description=run_name) as parent_run:
        for alpha in params_alpha:
            for l1_ratio in params_l1_ratio:
                k += 1
                print(f"\n***** ITERATION {k} from {num_iterations} *****")
                child_run_name = f"{run_name}_{k:02}"
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=RANDOM_STATE)
                with mlflow.start_run(run_name=child_run_name, nested=True) as child_run:
                    results = train_and_log_model(model, X_train, X_test, y_train, y_test)
                    # log hyperparameters
                    mlflow.log_param("alpha", alpha)
                    mlflow.log_param("l1_ratio", l1_ratio)
                    if results.metrics['root_mean_squared_error'] < best_score:
                        best_score = results.metrics['root_mean_squared_error']
                        best_run_id = child_run.info.run_id
                    print(f"rmse: {results.metrics['root_mean_squared_error']}")
                    print(f"r2: {results.metrics['r2_score']}")

    # Register the best model in the model registry
    model_uri = f"runs:/{best_run_id}/sklearn-model"
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    print("Model saved to the model registry:")
    print(f"Name: {mv.name}")
    print(f"Version: {mv.version}")
    print(f"Source: {mv.source}")

    # This is to show how to load models from the mlflow model registry
    print("#"*20)
    print("Load model from the model registry")
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    print(f"Model URI: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri=model_uri)
    y_pred = model.predict(X_test)
    print(f"RMSE for test data = {mean_squared_error(y_test, y_pred, squared=False)}")