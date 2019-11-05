import os
import json
import shutil
from .feature_engineering.vectorizer import log_to_vector, build_vocabulary
from .feature_engineering.utils import save_vocabulary, load_vocabulary
from .feature_engineering import registry as feature_registry
import numpy as np


# trim is only used when showing the top keywords for each class
def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


class TestingParameters():
    def __init__(self, params):
        self.params = params
        self.original_state = params['train']

    def __enter__(self):
        self.params['train'] = False

    def __exit__(self, exc_type, exc_value, traceback):
        self.params['train'] = self.original_state


def load_params(params):
    params_file = os.path.join(
        params['base_dir'], f"best_params.json")
    with open(params_file, "r") as fp:
        best_params = json.load(fp)
    params['experiment_id'] = best_params['experiment_id']
    params['features'] = best_params['features']
    params['healthy_label'] = best_params['healthy_label']


def save_params(params):
    params_file = os.path.join(
        params['base_dir'], f"best_params.json")
    with open(params_file, "w") as fp:
        json.dump(params, fp)


def file_handling(params):
    if params['train']:
        if os.path.exists(params["base_dir"]) and not params["force"]:
            raise FileExistsError(
                "directory '{} already exists. ".format(params["base_dir"])
                + "Run with --force to overwrite."
            )
        if os.path.exists(params["base_dir"]):
            shutil.rmtree(params["base_dir"])
        os.makedirs(params["base_dir"])
    else:
        if not os.path.exists(params["base_dir"]):
            raise FileNotFoundError(
                "directory '{} doesn't exist. ".format(params["base_dir"])
                + "Run train first before running inference."
            )


def get_features_vector(log_vector, vocabulary, params):
    feature_vectors = []
    for feature in params['features']:
        extract_feature = feature_registry.get_feature_extractor(feature)
        feature_vector = extract_feature(
            params, log_vector, vocabulary=vocabulary)
        feature_vectors.append(feature_vector)
    X = np.hstack(feature_vectors)
    return X


def extract_features(x, params):
    # Build Vocabulary
    if params['train']:
        vocabulary = build_vocabulary(x)
        save_vocabulary(params, vocabulary)
    else:
        vocabulary = load_vocabulary(params)
    # Feature Engineering
    x_vector = log_to_vector(x, vocabulary)
    x_features = get_features_vector(x_vector, vocabulary, params)
    return x_features, vocabulary
