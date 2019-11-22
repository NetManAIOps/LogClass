from .registry import register
from .vectorizer import (
    get_lf,
    calculate_ilf,
    calculate_tf_invf_train,
    create_invf_vector,
)
from .utils import save_feature_dict, load_feature_dict


@register("tfilf")
def create_tfilf_feature(params, train_vector, **kwargs):
    """
        Returns the tf-ilf matrix of features.
    """
    if params['train']:
        invf_dict = calculate_tf_invf_train(
            train_vector,
            get_f=get_lf,
            calc_invf=calculate_ilf
            )
        save_feature_dict(params, invf_dict, "tfilf")
    else:
        invf_dict = load_feature_dict(params, "tfilf")

    features = create_invf_vector(
        train_vector, invf_dict, kwargs['vocabulary'])
    return features
