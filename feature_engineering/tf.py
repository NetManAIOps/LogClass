from .registry import register
from .vectorizer import get_tf
import numpy as np
from .utils import save_feature_dict, load_feature_dict


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
        save_feature_dict(params, tf_dict, "tf")
    else:
        tf_dict = load_feature_dict(params, "tf")

    tf_features =\
        create_tf_vector(input_vector, tf_dict, kwargs['vocabulary'])

    return tf_features
