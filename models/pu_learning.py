from .binary_registry import register
from ..puLearning.puAdapter import PUAdapter
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel
import os
import pickle


class PUAdapterWrapper(BaseModel):
    def __init__(self, model, params):
        super().__init__(model, params)

    def save(self, **kwargs):
        pu_estimator_file = os.path.join(
            self.params['models_dir'],
            "pu_estimator.pkl"
            )
        pu_saver = {'estimator': self.model.estimator,
                    'c': self.model.c}
        with open(pu_estimator_file, 'wb') as pu_estimator_file:
            pickle.dump(pu_saver, pu_estimator_file)

    def load(self, **kwargs):
        pu_estimator_file = os.path.join(
            self.params['models_dir'],
            "pu_estimator.pkl"
            )
        with open(pu_estimator_file, 'rb') as pu_estimator_file:
            pu_saver = pickle.load(pu_estimator_file)
            estimator = pu_saver['estimator']
            pu_estimator = PUAdapter(estimator)
            pu_estimator.c = pu_saver['c']
            pu_estimator.estimator_fitted = True
            self.model = pu_estimator


@register("pu_learning")
def instatiate_pu_adapter(params, **kwargs):
    """
        Returns a RF adapted to do PU Learning wrapped by the PUAdapterWrapper.
    """
    hparms = {
        'n_estimators': 10,
        'criterion': "entropy",
        'bootstrap': True,
        'n_jobs': -1,
    }
    hparms.update(kwargs)
    estimator = RandomForestClassifier(**hparms)
    wrapped_pu_estimator = PUAdapterWrapper(PUAdapter(estimator), params)
    return wrapped_pu_estimator
