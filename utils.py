import numpy as np

# trim is only used when showing the top keywords for each class
def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

def addLengthInFeature(X, X_train_bag_vector):
    """
        X_train,max_length=addLengthForTrain(X_train,X_train_bag_vector)
        Return: X_train_with ,max_length
    """
    len_list = []
    for line in X_train_bag_vector:
        len_list.append(len(line))
    len_final = len(len_list)
    len_list = np.array(len_list).reshape(len_final, 1)
    X = np.hstack((X, len_list))
    return X