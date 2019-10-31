from .wb_registry import register
import numpy as np


def get_feature_names(params, vocabulary, add_length=True):
    feature_names = zip(vocabulary.keys(), vocabulary.values())
    feature_names = sorted(feature_names, key=lambda x: x[1])
    feature_names = [x[0] for x in feature_names]
    if 'length' in params['features']:
        feature_names.append('LENGTH')
    return np.array(feature_names)


@register('top_k_svm')
def get_top_k_SVM_features(params, model, vocabulary, **kwargs):
    hparms = {
        'target_names': [],
        'top_features': 5,
    }
    hparms.update(kwargs)

    top_k_label = {}
    feature_names = get_feature_names(params, vocabulary)
    for i, label in enumerate(hparms['target_names']):
        if len(hparms['target_names']) < 3 and i == 1:
            break  # coef is unidemensional when there's only two labels
        coef = model.coef_[i]
        top_coefficients = np.argsort(coef)[-hparms['top_features']:]
        top_k_features = feature_names[top_coefficients]
        top_k_label[label] = list(reversed(top_k_features))
    return top_k_label
