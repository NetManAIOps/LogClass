import numpy as np
from ..decorators import print_step
from collections import defaultdict, Counter


def get_ngrams(n, line):
    line = line.strip().split()
    cur_len = len(line)
    ngrams_list = []
    if cur_len == 0:
        # Token list is empty
        pass
    elif cur_len < n:
        # Token list fits in one ngram
        ngrams_list.append(" ".join(line))
    else:
        # Token list spans multiple ngrams
        loop_num = cur_len - n + 1
        for i in range(loop_num):
            cur_gram = " ".join(line[i: i + n])
            ngrams_list.append(cur_gram)
    return ngrams_list


def tokenize(line):
    return line.strip().split()


@print_step
def build_vocabulary(inputData):
    """ Divides log into tokens and creates vocabulary.

    Parameter
    ---------
    inputData: list of log message lines

    Returns
    -------
    vocabulary : word to index dict

    """
    vocabulary = {}
    for line in inputData:
        token_list = tokenize(line)
        for token in token_list:
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)
    return vocabulary


@print_step
def log_to_vector(inputData, vocabulary):
    """ Vectorizes each log message using a dict of words to index.

    Parameter
    ---------
    inputData: list of log message lines.
    vocabulary : word to index dict.

    Returns
    -------
    numpy Array vector of word indexes from each log message line.

    """
    result = []
    for line in inputData:
        temp = []
        token_list = tokenize(line)
        if token_list:
            for token in token_list:
                if token not in vocabulary:
                    continue
                else:
                    temp.append(vocabulary[token])
        result.append(temp)
    return np.array(result)


def setTrainDataForILF(x, y):
    x_res, indices = np.unique(x, return_index=True)
    y_res = y[indices]
    return x_res, y_res


def calculate_inv_freq(total, num):
    return np.log(float(total) / float(num + 0.01))


def get_max_line(inputVector):
    return len(max(inputVector, key=len))


def get_tf(inputVector):
    token_index_dict = defaultdict(set)
    # Counting the number of logs the word appears in
    for index, line in enumerate(inputVector):
        for token in line:
            token_index_dict[token].add(index)
    return token_index_dict


def get_lf(inputVector):
    token_index_ilf_dict = defaultdict(set)
    for line in inputVector:
        for location, token in enumerate(line):
            token_index_ilf_dict[token].add(location)
    return token_index_ilf_dict


def calculate_idf(token_index_dict, inputVector):
    idf_dict = {}
    total_log_num = len(inputVector)
    for token in token_index_dict:
        idf_dict[token] = calculate_inv_freq(total_log_num,
                                             len(token_index_dict[token]))
    return idf_dict


def calculate_ilf(token_index_dict, inputVector):
    ilf_dict = {}
    max_length = get_max_line(inputVector)
    # calculating ilf for each token
    for token in token_index_dict:
        ilf_dict[token] = calculate_inv_freq(max_length,
                                             len(token_index_dict[token]))
    return ilf_dict


def create_invf_vector(inputVector, invf_dict, vocabulary):
    tfinvf = []
    # Creating the idf/ilf vector for each log message
    for line in inputVector:
        cur_tfinvf = np.zeros(len(vocabulary))
        count_dict = Counter(line)
        for token_index in line:
            cur_tfinvf[token_index] = (
                float(count_dict[token_index]) * invf_dict[token_index]
            )
        tfinvf.append(cur_tfinvf)
    tfinvf = np.array(tfinvf)
    return tfinvf


def normalize_tfinvf(tfinvf):
    return 2.*(tfinvf - np.min(tfinvf))/np.ptp(tfinvf)-1


def calculate_tf_invf_train(
    inputVector, get_f=get_tf, calc_invf=calculate_idf
):
    token_index_dict = get_f(inputVector)
    invf_dict = calc_invf(token_index_dict, inputVector)
    return invf_dict
