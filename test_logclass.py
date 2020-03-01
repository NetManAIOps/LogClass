import pytest
from .feature_engineering.vectorizer import (
    build_vocabulary,
    log_to_vector,
    get_tf,
    get_lf,
    calculate_ilf,
    calculate_idf,
    create_invf_vector,
    )
from .preprocess import registry as preprocess_registry
import numpy as np


lines_input = [
        'Error writing message to the data stream',
        'Software detected a severe burst of similar RAS events and is halting There were similar events detected for message id',
    ]
vocab = {'Error': 0, 'writing': 1, 'message': 2, 'to': 3, 'the': 4, 'data': 5, 'stream': 6, 'Software': 7, 'detected': 8, 'a': 9, 'severe': 10, 'burst': 11, 'of': 12, 'similar': 13, 'RAS': 14, 'events': 15, 'and': 16, 'is': 17, 'halting': 18, 'There': 19, 'were': 20, 'for': 21, 'id': 22}
log_vectors = [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 13, 15, 8, 21, 2, 22]]
tf = {0: {0}, 1: {0}, 2: {0, 1}, 3: {0}, 4: {0}, 5: {0}, 6: {0}, 7: {1}, 8: {1}, 9: {1}, 10: {1}, 11: {1}, 12: {1}, 13: {1}, 14: {1}, 15: {1}, 16: {1}, 17: {1}, 18: {1}, 19: {1}, 20: {1}, 21: {1}, 22: {1}}
lf = {0: {0}, 1: {1}, 2: {2, 18}, 3: {3}, 4: {4}, 5: {5}, 6: {6}, 7: {0}, 8: {16, 1}, 9: {2}, 10: {3}, 11: {4}, 12: {5}, 13: {6, 14}, 14: {7}, 15: {8, 15}, 16: {9}, 17: {10}, 18: {11}, 19: {12}, 20: {13}, 21: {17}, 22: {19}}
lf_ilf = {0: 2.985781942700823, 1: 2.985781942700823, 2: 2.2975975514830065, 3: 2.985781942700823, 4: 2.985781942700823, 5: 2.985781942700823, 6: 2.985781942700823, 7: 2.985781942700823, 8: 2.2975975514830065, 9: 2.985781942700823, 10: 2.985781942700823, 11: 2.985781942700823, 12: 2.985781942700823, 13: 2.2975975514830065, 14: 2.985781942700823, 15: 2.2975975514830065, 16: 2.985781942700823, 17: 2.985781942700823, 18: 2.985781942700823, 19: 2.985781942700823, 20: 2.985781942700823, 21: 2.985781942700823, 22: 2.985781942700823}
tf_idf = {0: 0.6831968497067772, 1: 0.6831968497067772, 2: -0.004987541511038939, 3: 0.6831968497067772, 4: 0.6831968497067772, 5: 0.6831968497067772, 6: 0.6831968497067772, 7: 0.6831968497067772, 8: 0.6831968497067772, 9: 0.6831968497067772, 10: 0.6831968497067772, 11: 0.6831968497067772, 12: 0.6831968497067772, 13: 0.6831968497067772, 14: 0.6831968497067772, 15: 0.6831968497067772, 16: 0.6831968497067772, 17: 0.6831968497067772, 18: 0.6831968497067772, 19: 0.6831968497067772, 20: 0.6831968497067772, 21: 0.6831968497067772, 22: 0.6831968497067772}
invf_dict_tfidf = [[0.6831968497067772, 0.6831968497067772, -0.004987541511038939, 0.6831968497067772, 0.6831968497067772, 0.6831968497067772, 0.6831968497067772, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -0.004987541511038939, 0.0, 0.0, 0.0, 0.0, 0.6831968497067772, 1.3663936994135544, 0.6831968497067772, 0.6831968497067772, 0.6831968497067772, 0.6831968497067772, 1.3663936994135544, 0.6831968497067772, 1.3663936994135544, 0.6831968497067772, 0.6831968497067772, 0.6831968497067772, 0.6831968497067772, 0.6831968497067772, 0.6831968497067772, 0.6831968497067772]]
invf_dict_lfilf = [[2.985781942700823, 2.985781942700823, 2.2975975514830065, 2.985781942700823, 2.985781942700823, 2.985781942700823, 2.985781942700823, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.2975975514830065, 0.0, 0.0, 0.0, 0.0, 2.985781942700823, 4.595195102966013, 2.985781942700823, 2.985781942700823, 2.985781942700823, 2.985781942700823, 4.595195102966013, 2.985781942700823, 4.595195102966013, 2.985781942700823, 2.985781942700823, 2.985781942700823, 2.985781942700823, 2.985781942700823, 2.985781942700823, 2.985781942700823]]


@pytest.mark.parametrize("test_input,expected", [(lines_input, vocab)])
def test_build_vocabulary(test_input, expected):
    assert build_vocabulary(test_input) == expected


@pytest.mark.parametrize("test_input,vocab,expected", [(lines_input, vocab, log_vectors)])
def test_log_to_vector(test_input, vocab, expected):
    assert all(log_to_vector(test_input, vocab) == expected)


@pytest.mark.skip(reason="Not working on this step currently")
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


@pytest.mark.parametrize("test_input,expected", [(np.array(log_vectors), tf)])
def test_get_tf(test_input, expected):
    result = get_tf(test_input)
    assert result == expected


@pytest.mark.parametrize("test_input,expected", [(np.array(log_vectors), lf)])
def test_get_lf(test_input, expected):
    result = get_lf(test_input)
    assert result == expected


@pytest.mark.parametrize("token_index_input_dict,input_vector,expected", [(lf, np.array(log_vectors), lf_ilf)])
def test_calculate_ilf(token_index_input_dict, input_vector, expected):
    result = calculate_ilf(token_index_input_dict, input_vector)
    assert result == expected


@pytest.mark.parametrize("token_index_input_dict,input_vector,expected", [(tf, np.array(log_vectors), tf_idf)])
def test_calculate_idf(token_index_input_dict, input_vector, expected):
    result = calculate_idf(token_index_input_dict, input_vector)
    assert result == expected


@pytest.mark.parametrize("input_vector,invf_dict,vocab,expected", [(np.array(log_vectors), tf_idf, vocab, invf_dict_tfidf), (np.array(log_vectors), lf_ilf, vocab, invf_dict_lfilf)])
def test_create_invf_vector(input_vector, invf_dict, vocab, expected):
    result = create_invf_vector(input_vector, invf_dict, vocab).tolist()
    assert result == expected
