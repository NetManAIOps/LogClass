"""Registry for black box reports or metrics."""

_BB_REPORTS = dict()


def register(name):
    """Registers a new black box report or metric function."""

    def add_to_dict(func):
        _BB_REPORTS[name] = func
        return func

    return add_to_dict


def get_bb_report(model):
    """Fetches the black box report or metric function."""
    return _BB_REPORTS[model]
