import os
import json
import shutil
from .feature_engineering.vectorizer import log_to_vector, build_vocabulary
from .feature_engineering.utils import save_vocabulary, load_vocabulary
from .feature_engineering import registry as feature_registry
import numpy as np
from .decorators import print_step
import pandas as pd


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
        params['id_dir'], f"best_params.json")
    with open(params_file, "r") as fp:
        best_params = json.load(fp)
    params.update(best_params)


def save_params(params):
    params_file = os.path.join(
        params['id_dir'], f"best_params.json")
    with open(params_file, "w") as fp:
        json.dump(params, fp)


def file_handling(params):
    if params['preprocess']:
        if not os.path.exists(params['raw_logs']):
            raise FileNotFoundError(
                f"File {params['raw_logs']} doesn't exist. "
                + "Please provide the raw logs path."
            )
        if not os.path.exists(params['logs']):
            directory = os.path.dirname(params['logs'])
            os.makedirs(directory)
    else:
        # Checks if preprocessed logs exist as input
        if not os.path.exists(params['logs']):
            raise FileNotFoundError(
                f"File {params['base_dir']} doesn't exist. "
                + "Preprocess target logs first and provide their path."
            )

    if params['train']:
        # Checks if the experiment id already exists
        if os.path.exists(params["id_dir"]) and not params["force"]:
            raise FileExistsError(
                f"directory '{params['id_dir']} already exists. "
                + "Run with --force to overwrite."
                + f"If --force is used, you could lose your training results."
            )
        if os.path.exists(params["id_dir"]):
            shutil.rmtree(params["id_dir"])
        for target_dir in ['id_dir', 'models_dir', 'features_dir']:
            os.makedirs(params[target_dir])
    else:
        # Checks if input models and features are provided
        for concern in ['models_dir', 'features_dir']:
            target_path = params[concern]
            if not os.path.exists(target_path):
                raise FileNotFoundError(
                    "directory '{} doesn't exist. ".format(target_path)
                    + "Run train first before running inference."
                )


@print_step
def get_features_vector(log_vector, vocabulary, params):
    feature_vectors = []
    for feature in params['features']:
        extract_feature = feature_registry.get_feature_extractor(feature)
        feature_vector = extract_feature(
            params, log_vector, vocabulary=vocabulary)
        feature_vectors.append(feature_vector)
    X = np.hstack(feature_vectors)
    return X


@print_step
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


def print_params(params):
    print("{:-^80}".format("params"))
    print("Beginning binary classification "
          + "using the following configuration:\n")
    for param, value in params.items():
        print("\t{:>13}: {}".format(param, value))
    print()
    print("-" * 80)


def save_results(results, params):
    df = pd.DataFrame(results)
    file_name = os.path.join(
        params['id_dir'],
        "results.csv",
        )
    df.to_csv(file_name, index=False)
