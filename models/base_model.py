from abc import ABC, abstractmethod
from time import time
from ..decorators import print_step


class BaseModel(ABC):
    """ Abstract class used to wrap models and add further functionality.

        Attributes
        ----------
        model : model that implements fit and predict functions as sklearn
        ML models do.
        params : dict of experiment parameters.
        name : str of the original model class name.
        train_time : time it took to run fit in seconds.
        run_time : time it took to run predict in seconds.

        Methods
        -------
        save(self, **kwargs)
            Abstract method for the subclass to implement how the model is
            saved. Should use the experiment id as reference.
        load(self, **kwargs)
            Abstract method for the subclass to implement how it's meant to be
            loaded. Should correspond to how the save method saves the model.
        predict(self, X, **kwargs)
            Wraps original model predict and times its running time.
        fit(self, X, Y, **kwargs)
            Wraps original model fit, times fit running time and saves the model.

    """
    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.name = type(model).__name__
        self.train_time = None
        self.run_time = None

    @abstractmethod
    def save(self, **kwargs):
        """
            Abstract method for the subclass to implement how the model is
            saved. Should use the experiment id as reference.
        """
        pass

    @abstractmethod
    def load(self, **kwargs):
        """
            Abstract method for the subclass to implement how it's meant to be
            loaded. Should correspond to how the save method saves the model.
        """
        pass

    @print_step
    def predict(self, X, **kwargs):
        """
            Wraps original model predict and times its running time.
        """
        t0 = time()
        pred = self.model.predict(X, **kwargs)
        t1 = time()
        lapse = t1 - t0
        self.run_time = lapse
        print(f"{self.name} took {lapse}s to run inference.")
        return pred

    @print_step
    def fit(self, X, Y, **kwargs):
        """
            Wraps original model fit, times fit running time and saves the model.
        """
        t0 = time()
        self.model.fit(X, Y, **kwargs)
        t1 = time()
        lapse = t1 - t0
        self.train_time = lapse
        print(f"{self.name} took {lapse}s to train.")
        self.save()
