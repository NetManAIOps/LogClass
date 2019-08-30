import numpy as np


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


def build_vocabulary(inputData):
    """
        Divides log into tokens and creates vocabulary.
        Args:
            inputData: list of log lines
        Returns:
            Vocabulary to index dict
    """
    vocabulary = {}
    for line in inputData:
        token_list = tokenize(line)
        for token in token_list:
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)
    return vocabulary


def log_to_vector(inputData, vocabulary, y):
    result = []
    y_result = []
    for index_data, line in enumerate(inputData):
        temp = []
        token_list = tokenize(line)
        if token_list:
            for token in token_list:
                if token not in vocabulary:
                    continue
                else:
                    temp.append(vocabulary[token])
        result.append(temp)
        y_result.append(y[index_data])
    return np.array(result), np.array(y_result)


def setTrainDataForILF(x, y):
    x_res, indices = np.unique(x, return_index=True)
    y_res = y[indices]
    return x_res, y_res


def calculate_inv_freq(total, num):
    return np.log(float(total) / float(num + 0.01))


def get_max_line(inputVector):
    return len(max(inputVector, key=len))


def get_tf(inputVector):
    token_index_dict = {}
    # Counting the number of logs the word appears in
    for index, line in enumerate(inputVector):
        for token in line:
            if token not in token_index_dict:
                token_index_dict[token] = set()
            token_index_dict[token].add(index)
    return token_index_dict


def get_lf(inputVector):
    token_index_ilf_dict = {}
    for line in inputVector:
        for location, token in enumerate(line):
            if token not in token_index_ilf_dict:
                token_index_ilf_dict[token] = set()
            token_index_ilf_dict[token].add(location)
    return token_index_ilf_dict


def calculate_idf(token_index_dict, inputVector, vocabulary):
    idf_dict = {}
    total_log_num = len(inputVector)
    for token in token_index_dict:
        idf_dict[token] = calculate_inv_freq(total_log_num,
                                             len(token_index_dict[token]))
    return idf_dict


def calculate_ilf(token_index_dict, inputVector, vocabulary):
    ilf_dict = {}
    max_length = get_max_line(inputVector)
    # calculating ilf for each token
    for token in token_index_dict:
        ilf_dict[token] = calculate_inv_freq(max_length,
                                             len(token_index_dict[token]))
    return ilf_dict


def create_invf_vector(invf_dict, inputVector, vocabulary):
    tfinvf = []
    # Creating the idf/ilf vector for each log message
    for line in inputVector:
        cur_tfinvf = np.zeros(len(vocabulary))
        for token_index in line:
            cur_tfinvf[token_index] = (
                float(line.count(token_index)) * invf_dict[token_index]
            )
        tfinvf.append(cur_tfinvf)

    tfinvf = np.array(tfinvf)
    return tfinvf


def normalize_tfinvf(tfinvf):
    return 2.*(tfinvf - np.min(tfinvf))/np.ptp(tfinvf)-1


def calculate_tf_invf_train(
    inputVector, vocabulary, get_f=get_tf, calc_invf=calculate_idf
):
    token_index_dict = get_tf(inputVector)
    invf_dict = calc_invf(token_index_dict, inputVector, vocabulary)
    tfinvf = create_invf_vector(invf_dict, inputVector, vocabulary)
    return tfinvf, invf_dict
