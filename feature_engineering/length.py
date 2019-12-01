from .registry import register
import numpy as np


@register("length")
def create_length_feature(params, input_vector, **kwargs):
    """
        Returns an array of lengths of each tokenized log message from the input.

        Parameters
        ----------
        params : dict of experiment parameters.
        input_vector : numpy Array vector of word indexes from each log message line.

        Returns
        -------
        numpy array of lengths of each tokenized log message from the input
        with shape (number_of_logs, N).
    """
    length = np.vectorize(len)
    length_feature = length(input_vector)
    length_feature = length_feature.reshape(-1, 1)
    return length_feature
