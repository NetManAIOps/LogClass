from .registry import register
from .vectorizer import (
    get_tf,
    calculate_idf,
    calculate_tf_invf_train,
    create_invf_vector,
)
from .utils import save_invf, load_invf


@register("tfidf")
def create_tfidf_feature(params, train_vector, **kwargs):
    """
        Returns the tf-idf matrix of features.
    """
    if params['train']:
        invf_dict = calculate_tf_invf_train(
            train_vector,
            get_f=get_tf,
            calc_invf=calculate_idf
            )
        save_invf(params, invf_dict)
    else:
        invf_dict = load_invf(params)

    features = create_invf_vector(
        train_vector, invf_dict, kwargs['vocabulary'])
    return features
