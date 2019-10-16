from .registry import register
import numpy as np


@register("length")
def create_length_feature(params, X_train_bag_vector, **kwargs):
    """
        Returns an array of the length of each tokenized log message from the input.
    """
    length = np.vectorize(len)
    length_feature = length(X_train_bag_vector)
    length_feature = length_feature.reshape(-1, 1)
    return length_feature
