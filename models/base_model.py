from abc import ABC, abstractmethod
from time import time
from ..decorators import print_step


class BaseModel(ABC):
    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.name = type(model).__name__
        self.train_time = None
        self.run_time = None

    @abstractmethod
    def save(self, **kwargs):
        pass

    @abstractmethod
    def load(self, **kwargs):
        pass

    @print_step
    def predict(self, X, **kwargs):
        t0 = time()
        pred = self.model.predict(X, **kwargs)
        t1 = time()
        lapse = t1 - t0
        self.run_time = lapse
        print(f"{self.name} took {lapse}s to run inference.")
        return pred

    @print_step
    def fit(self, X, Y, **kwargs):
        t0 = time()
        self.model.fit(X, Y, **kwargs)
        t1 = time()
        lapse = t1 - t0
        self.train_time = lapse
        print(f"{self.name} took {lapse}s to train.")
        self.save()
