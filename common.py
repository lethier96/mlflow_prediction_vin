import os
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# full config path
CONFIG_PATH = os.path.join(ROOT_DIR, 'config.yml')

def get_full_path(rel_path):
    return os.path.normpath(os.path.join(ROOT_DIR, rel_path))

with open(CONFIG_PATH, "r") as f:

    # yaml to python dict
    CONFIG = yaml.load(f, Loader=yaml.SafeLoader)

    for key, value in CONFIG['paths'].items():
        CONFIG['paths'][key] = get_full_path(value)

def calc_scores_linearmodel(actual, pred):

    rmse = mean_squared_error(actual, pred, squared=False)
    r2 = r2_score(actual, pred)

    return {
        "rmse": rmse,
        "r2": r2}