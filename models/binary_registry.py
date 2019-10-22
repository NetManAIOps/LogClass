"""Registry for binary models to be used for anomaly detection."""

_BINARY_MODELS = dict()


def register(name):
    """Registers a new binary classification anomaly detection model."""

    def add_to_dict(func):
        _BINARY_MODELS[name] = func
        return func

    return add_to_dict


def get_binary_model(model):
    """Fetches the binary classification anomaly detection model."""
    return _BINARY_MODELS[model]
