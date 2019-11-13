from .registry import register
from .vectorizer import (
    get_tf,
)
import pickle
import os
import numpy as np


def save_tf(params, tf_dict):
    tf_dict_file = os.path.join(
        params['base_dir'], f"tf_dict_{params['experiment_id']}.pkl")
    with open(tf_dict_file, "wb") as fp:
        pickle.dump(tf_dict, fp)


def load_tf(params):
    tf_dict_file = os.path.join(
        params['base_dir'], f"tf_dict_{params['experiment_id']}.pkl")
    with open(tf_dict_file, "rb") as fp:
        tf_dict = pickle.load(fp)
    return tf_dict


def create_tf_vector(input_vector, tf_dict, vocabulary):
    tf_vector = []
    # Creating the idf/ilf vector for each log message
    for line in input_vector:
        cur_tf_vector = np.zeros(len(vocabulary))
        for token_index in line:
            cur_tf_vector[token_index] = len(tf_dict[token_index])
        tf_vector.append(cur_tf_vector)

    tf_vector = np.array(tf_vector)
    return tf_vector


@register("tf")
def create_term_count_feature(params, input_vector, **kwargs):
    """
        Returns an array of the counts of each word per log message.
    """
    if params['train']:
        tf_dict = get_tf(input_vector)
        save_tf(params, tf_dict)
    else:
        tf_dict = load_tf(params)

    tf_features =\
        create_tf_vector(input_vector, tf_dict, kwargs['vocabulary'])

    return tf_features
