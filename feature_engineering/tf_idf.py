from .registry import register
from .vectorizer import (
    get_tf,
    calculate_idf,
    calculate_tf_invf_train,
    create_invf_vector,
)
from .utils import save_feature_dict, load_feature_dict


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
        save_feature_dict(params, invf_dict, "tfidf")
    else:
        invf_dict = load_feature_dict(params, "tfidf")

    features = create_invf_vector(
        train_vector, invf_dict, kwargs['vocabulary'])
    return features
