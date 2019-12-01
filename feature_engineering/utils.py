import os
import pickle
import numpy as np
from .vectorizer import log_to_vector, build_vocabulary
from . import registry as feature_registry
from ..decorators import print_step


def load_feature_dict(params, name):
    dict_file = os.path.join(params['features_dir'], f"{name}.pkl")
    with open(dict_file, "rb") as fp:
        feat_dict = pickle.load(fp)
    return feat_dict


def save_feature_dict(params, feat_dict, name):
    dict_file = os.path.join(params['features_dir'], f"{name}.pkl")
    with open(dict_file, "wb") as fp:
        pickle.dump(feat_dict, fp)


def binary_train_gtruth(y):
    return np.where(y == -1.0, -1.0, 1.0)


def multi_features(x, y):
    anomalous = (y != -1)
    x_multi, y_multi = x[anomalous], y[anomalous]
    return x_multi, y_multi


@print_step
def get_features_vector(log_vector, vocabulary, params):
    """ Extracts all specified features from the vectorized logs.

    For each feature specified in params it gets the feature function from the
    feature registry and applies to the data.
    A numpy array vector of shape (number_of_logs, N) is expected for each to
    be concatenated along the second axis.

    Parameters
    ----------
    log_vector : numpy Array vector of word indexes from each log message line.
    vocabulary : dict mapping a word to an index.
    params : dict of experiment parameters.

    Returns
    -------
    x_features : numpy ndArray of all specified features.

    """
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
    """ Gets vocabulary and specified features from the preprocessed logs.

    Creates a vocabulary from the preprocessed logs to vectorize each message.
    Extracts all specified features in params from the logs vector and
    vocabulary, then returns them both.

    Parameters
    ----------
    x : list of preprocessed logs. One log message per line.
    params : dict of experiment parameters.

    Returns
    -------
    x_features : numpy ndArray of all specified features.
    vocabulary : dict mapping a word to an index.

    """
    # Build Vocabulary
    if params['train']:
        vocabulary = build_vocabulary(x)
        save_feature_dict(params, vocabulary, "vocab")
    else:
        vocabulary = load_feature_dict(params, "vocab")
    # Feature Engineering
    x_vector = log_to_vector(x, vocabulary)
    x_features = get_features_vector(x_vector, vocabulary, params)
    return x_features, vocabulary
