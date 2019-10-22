"""Registry for multi-class models to be used for anomaly classification."""

_MULTI_MODELS = dict()


def register(name):
    """Registers a new multi-class anomaly classification model."""

    def add_to_dict(func):
        _MULTI_MODELS[name] = func
        return func

    return add_to_dict


def get_multi_model(model):
    """Fetches the multi-class anomaly classification model."""
    return _MULTI_MODELS[model]
