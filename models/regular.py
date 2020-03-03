from .binary_registry import register
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel
import os
import pickle


class RegularClassifierWrapper(BaseModel):
    def __init__(self, model, params):
        super().__init__(model, params)

    def save(self, **kwargs):
        regular_file = os.path.join(
            self.params['models_dir'],
            "regular.pkl"
            )
        with open(regular_file, 'wb') as regular_clf_file:
            pickle.dump(self.model, regular_clf_file)

    def load(self, **kwargs):
        regular_file = os.path.join(
            self.params['models_dir'],
            "regular.pkl"
            )
        with open(regular_file, 'rb') as regular_clf_file:
            regular_classifier = pickle.load(regular_clf_file)
            self.model = regular_classifier


@register("regular")
def instatiate_regular_classifier(params, **kwargs):
    """
        Returns a RF wrapped by the PU Learning Adapter.
    """
    hparms = {
        'n_estimators': 10,
        'bootstrap': True,
        'n_jobs': -1,
    }
    hparms.update(kwargs)
    wrapped_regular = RegularClassifierWrapper(
        RandomForestClassifier(**hparms), params)
    return wrapped_regular
