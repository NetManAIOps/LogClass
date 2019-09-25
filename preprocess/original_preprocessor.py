from .registry import register
from .utils import getMsgFromNewSyslog


@register("original")
def preprocess_original():
    """
    Returns preprocessed original logs from the paper.
    """
    return load_from_keras(
        tf.keras.datasets.mnist, num_valid, label_smoothing=label_smoothing
    )