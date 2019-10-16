"""Basic registry for logs vector feature engineering. These take the
log messages as input and extract and return a feature to be appended to
the feature vector."""

_FEATURE_EXTRACTORS = dict()


def register(name):
    """Registers a new log message feature extraction function under the
    given name."""

    def add_to_dict(func):
        _FEATURE_EXTRACTORS[name] = func
        return func

    return add_to_dict


def get_feature_extractor(feature):
    """Fetches the feature extraction function associated with the given
    raw logs"""
    return _FEATURE_EXTRACTORS[feature]
