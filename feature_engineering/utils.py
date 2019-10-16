import json
import os
import pickle
import numpy as np


def load_invf(params):
    invf_dict_file = os.path.join(
        params['base_dir'], f"invf_dict_{params['experiment_id']}.pkl")
    with open(invf_dict_file, "rb") as fp:
        invf_dict = pickle.load(fp)
    return invf_dict


def save_invf(params, invf_dict):
    invf_dict_file = os.path.join(
        params['base_dir'], f"invf_dict_{params['experiment_id']}.pkl")
    with open(invf_dict_file, "wb") as fp:
        pickle.dump(invf_dict, fp)


def load_vocabulary(params):
    vocab_file = os.path.join(
        params['base_dir'], f"vocab_{params['experiment_id']}.json")
    with open(vocab_file, "r") as fp:
        vocabulary = json.load(fp)
    return vocabulary


def save_vocabulary(params, vocabulary):
    vocab_file = os.path.join(
        params['base_dir'], f"vocab_{params['experiment_id']}.json")
    with open(vocab_file, "w") as fp:
        json.dump(vocabulary, fp)


def binary_train_gtruth(y):
    return np.where(y == -1.0, -1.0, 1.0)


def multi_class_gtruth(x, y):
    anomalous = (y != -1)
    x_multi, y_multi =\
        x[anomalous], y[anomalous]
    return x_multi, y_multi
