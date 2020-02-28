import pytest
from .feature_engineering.vectorizer import (
    build_vocabulary,
    log_to_vector,
    )
from .preprocess import registry as preprocess_registry


lines_input = [
        'Error writing message to the data stream',
        'Software detected a severe burst of similar RAS events and is halting There were similar events detected for message id',
    ]
vocab = {'Error': 0, 'writing': 1, 'message': 2, 'to': 3, 'the': 4, 'data': 5, 'stream': 6, 'Software': 7, 'detected': 8, 'a': 9, 'severe': 10, 'burst': 11, 'of': 12, 'similar': 13, 'RAS': 14, 'events': 15, 'and': 16, 'is': 17, 'halting': 18, 'There': 19, 'were': 20, 'for': 21, 'id': 22}
log_vectors = [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 13, 15, 8, 21, 2, 22]]

@pytest.mark.parametrize("test_input,expected", [(lines_input, vocab)])
def test_build_vocabulary(test_input, expected):
    assert build_vocabulary(test_input) == expected


@pytest.mark.parametrize("test_input,vocab,expected", [(lines_input, vocab, log_vectors)])
def test_log_to_vector(test_input, vocab, expected):
    assert all(log_to_vector(test_input, vocab) == expected)


def test_bgl_preprocessing():
    import os
    import hashlib
    # TODO: get checksum directly from the file name instead
    checksum = 'de09099fb4457d819b53ab60453da974'
    logs_type = 'bgl'
    params = {
        'raw_logs': os.path.normpath(f'./data/fixtures/{checksum}'),
        'logs': os.path.normpath(f'./data/fixtures/{logs_type}'),
        'logs_type': logs_type,
    }
    preprocess = preprocess_registry.get_preprocessor(params['logs_type'])
    preprocess(params)
    with open(params['logs'], "rb") as f:
        file_check = f.read()
        file_md5 = hashlib.md5(file_check).hexdigest()
    assert file_md5 == checksum


