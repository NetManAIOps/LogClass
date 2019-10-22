from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, model, params):
        self.model = model
        self.params = params

    @abstractmethod
    def save(self, **kwargs):
        pass

    @abstractmethod
    def load(self, **kwargs):
        pass

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def fit(self, X, Y, **kwargs):
        self.model.fit(X, Y, **kwargs)
        self.save()
