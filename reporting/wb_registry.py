"""Registry for white box reports or metrics."""

_WB_REPORTS = dict()


def register(name):
    """Registers a new white box report or metric function."""

    def add_to_dict(func):
        _WB_REPORTS[name] = func
        return func

    return add_to_dict


def get_wb_report(model):
    """Fetches the white box report or metric function."""
    return _WB_REPORTS[model]
