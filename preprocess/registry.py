"""Basic registry for logs preprocessors. These read the rawlog file and
outputs filtered logs removing parameter words or tokens with non-letter
characters keeping only text words."""

_PREPROCESSORS = dict()


def register(name):
    """Registers a new logs preprocessor function under the given name."""

    def add_to_dict(func):
        _PREPROCESSORS[name] = func
        return func

    return add_to_dict


def get_preprocessor(data_src):
    """Fetches the logs preprocessor function associated with the given raw logs"""
    return _PREPROCESSORS[data_src]
