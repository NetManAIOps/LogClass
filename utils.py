# trim is only used when showing the top keywords for each class
def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


class TestingParameters():
    def __init__(self, params):
        self.params = params
        self.original_state = params['train']

    def __enter__(self):
        self.params['train'] = False

    def __exit__(self, exc_type, exc_value, traceback):
        self.params['train'] = self.original_state
