import os
import json


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


def load_params(params):
    params_file = os.path.join(
        params['base_dir'], f"best_params.json")
    with open(params_file, "r") as fp:
        best_params = json.load(fp)
    params['experiment_id'] = best_params['experiment_id']
    params['features'] = best_params['features']
    params['healthy_label'] = best_params['healthy_label']


def save_params(params):
    params_file = os.path.join(
        params['base_dir'], f"best_params.json")
    with open(params_file, "w") as fp:
        json.dump(params, fp)
