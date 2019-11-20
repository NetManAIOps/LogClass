from .multi_registry import register
from sklearn.svm import LinearSVC
from .base_model import BaseModel
import os
import pickle


class SVMWrapper(BaseModel):
    def __init__(self, model, params):
        super().__init__(model, params)

    def save(self, **kwargs):
        multi_file = os.path.join(
            self.params['models_dir'],
            "multi.pkl"
            )
        with open(multi_file, 'wb') as multi_clf_file:
            pickle.dump(self.model, multi_clf_file)

    def load(self, **kwargs):
        multi_file = os.path.join(
            self.params['models_dir'],
            "multi.pkl"
            )
        with open(multi_file, 'rb') as multi_clf_file:
            multi_classifier = pickle.load(multi_clf_file)
            self.model = multi_classifier


@register("svm")
def instatiate_svm(params, **kwargs):
    """
        Returns a RF wrapped by the PU Learning Adapter.
    """
    hparms = {
        'penalty': "l2",
        'dual': False,
        'tol': 1e-1,
    }
    hparms.update(kwargs)
    wrapped_svm = SVMWrapper(LinearSVC(**hparms), params)
    return wrapped_svm
