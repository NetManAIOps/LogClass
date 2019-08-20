# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : Weibin Meng
# * Email         : m_weibin@163.com
# * Create time   : 2017-09-06 16:48
# * Last modified : 2019-08-05 15:40
# * Filename      : total_mululti-logClassification.py
# * Description   :
"""
ilf is faster than idf

Each gram is for each gram, not multi of words.
This version can save label+result+log
If add length feature, --add_length
"""
# **********************************************************

from __future__ import print_function
from sklearn.metrics import f1_score
import sys
import matplotlib

matplotlib.use("Agg")
import logging
import numpy as np
import math
from optparse import OptionParser
from time import time
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
import sys

# input_path='./data/logs_without_paras.txt'
# k_of_kflod=10
n_for_gram = 1
total_tol = 1e-1  # param of svc


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# parse commandline arguments
op = OptionParser()
op.add_option(
    "--add_length",
    action="store_true",
    dest="add_length_vector",
    help="Add length vector",
)
op.add_option(
    "--logs",
    action="store",
    type="string",
    dest="logs",
    default="./data/logs_without_paras.txt",
    help="Input filename",
)
op.add_option(
    "--kfold",
    action="store",
    type="int",
    dest="k_of_kflod",
    default=10,
    help="k of k-flod crossvalidation",
)
op.add_option(
    "--report",
    action="store_true",
    dest="print_report",
    help="Print a detailed classification report.",
)
op.add_option(
    "--chi2_select",
    action="store",
    type="int",
    dest="select_chi2",
    help="Select some number of features using a chi-squared test",
)
op.add_option(
    "--confusion_matrix",
    action="store_true",
    dest="print_cm",
    help="Print the confusion matrix.",
)
op.add_option(
    "--top10",
    action="store_true",
    dest="print_top10",
    help="Print ten most discriminative terms per class" " for every classifier.",
)
op.add_option(
    "--all_categories",
    action="store_true",
    dest="all_categories",
    help="Whether to use all categories or not.",
)
op.add_option("--add_ilf", action="store_true", dest="add_ilf", help="add_ilf")
op.add_option(
    "--less_train", action="store_true", dest="less_train", help="less train data"
)
op.add_option(
    "--n_features",
    action="store",
    type=int,
    default=2 ** 16,
    help="n_features when using the hashing vectorizer.",
)
op.add_option(
    "--filtered",
    action="store_true",
    help="Remove newsgroup information that is easily overfit: "
    "headers, signatures, and quoting.",
)


def is_interactive():
    return not hasattr(sys.modules["__main__"], "__file__")


# trim is only used when showing the top keywords for each class
def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

def get_ngrams(n, line):
    line = line.strip().split()
    cur_len = len(line)
    cur_index = 0
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
            cur_gram = " ".join(line[i : i + n])
            ngrams_list.append(cur_gram)
    return ngrams_list

def build_ngram_vocabulary(n, inputData):
    """
        Divides log into n-gram using get_ngrams method and creates vocabulary.
        Args:
            n: ngram size
            inputData: list of log lines
        Returns: 
            Vocabulary to index dict
    """
    vocabulary = {}
    for line in inputData:
        ngrams_list = get_ngrams(n, line)
        for ngram in ngrams_list:
            if not ngram in vocabulary:
                vocabulary[ngram] = len(vocabulary)

    return vocabulary

def log_to_vector(n, inputData, vocabulary, y):
    result = []
    x_result = []
    y_result = []
    for index_data, line in enumerate(inputData):
        temp = []
        ngrams_list = get_ngrams(n, line)
        if ngrams_list:
            for cur_gram in ngrams_list:
                if cur_gram not in vocabulary:
                    continue
                else:
                    temp.append(vocabulary[cur_gram]) 
        result.append(temp)
        x_result.append(line)
        y_result.append(y[index_data])
    return np.array(result), np.array(y_result), x_result


def setTrainDataForILF(x, y):
    x_res, indices = np.unique(x, return_index=True)
    y_res = y[indices]

    return x_res, y_res

def calculate_inv_freq(total, num):
    return np.log(
            float(total) / float(num + 0.01)
        )

# CHECK IF USING MAX IS BETTER OR SIMILAR SPEED AS IT'S EASIER TO READ
# WHAT ABOUT NUMPY?
def get_max_line(inputVector):
    max_length = 0
    for line in inputVector:
        if len(line) > max_length:
            max_length = len(line)
    return max_length

def get_tf(inputVector):
    gram_index_dict = {}
    # Counting the number of logs the word appears in
    for index, line in enumerate(inputVector):
        for gram in line:
            if gram not in gram_index_dict:
                gram_index_dict[gram] = set()
            gram_index_dict[gram].add(index)
    return gram_index_dict

def get_lf(inputVector):
    gram_index_ilf_dict = {}
    for line in inputVector:
        for location, gram in enumerate(line):
            if gram not in gram_index_ilf_dict:
                gram_index_ilf_dict[gram] = set()
            gram_index_ilf_dict[gram].add(location)
    return gram_index_ilf_dict

def calculate_idf(gram_index_dict, inputVector, vocabulary):
    idf_dict = {}
    total_log_num = len(inputVector)
    for index, gram in enumerate(gram_index_dict):
            idf_dict[gram] = calculate_inv_freq(total_log_num, len(gram_index_dict[gram]))
    return idf_dict

def calculate_ilf(gram_index_dict, inputVector, vocabulary):
    ilf_dict = {}
    max_length = get_max_line(inputVector)
    # calculating ilf for each gram
    for index, gram in enumerate(gram_index_dict):
        ilf_dict[gram] = calculate_inv_freq(max_length, len(gram_index_dict[gram]))
    return ilf_dict

def create_invf_vector(invf_dict, inputVector, vocabulary):
    tfinvf = []
    # Creating the idf/ilf vector for each log message
    for index, line in enumerate(inputVector):
        cur_tfinvf = np.zeros(len(vocabulary))
        for gram_index in line:
                cur_tfinvf[gram_index] = (
                    float(line.count(gram_index)) * invf_dict[gram_index]
                )
        tfinvf.append(cur_tfinvf)

    tfinvf = np.array(tfinvf)
    return tfinvf

def calculate_tf_invf_train(inputVector, vocabulary, get_tf=get_tf, calc_invf=calculate_idf):
    """
        In this version, tf is not normalized. We use frequence value as tf value.

            RETURN: tfidf,tfidf_mean,tfidf_std,idf_dict
    """
    gram_index_dict = get_tf(inputVector)
    invf_dict = calc_invf(gram_index_dict, inputVector, vocabulary)
    tfinvf = create_invf_vector(invf_dict, inputVector, vocabulary)
    return tfinvf, invf_dict

def calculate_tf_invf_test(invf_dict, inputVector, vocabulary):
    return create_invf_vector(invf_dict, inputVector, vocabulary)

# I'd separate idf from ilf into different methods
# Same for tf into a different method that is called
# Also have a method for the tf-idf math formula to make it legible
# Maybe also another method for the third block that builds the vector (let's do it as it's exactly what the tdif for test does)
def calculateTfidfForTrain(inputVector, vocabulary):
    """
        In this version, tf is not normalized. We use frequence value as tf value.

            RETURN: tfidf,tfidf_mean,tfidf_std,idf_dict
    """
    # gram_index_ilf_dict = {}
    # gram_index_dict = {}
    # max_length = 0
    # # Finding max log message length
    # for index, line in enumerate(inputVector):
    #     if len(line) > max_length:
    #         max_length = len(line)
    #     for location, gram in enumerate(line):
    #         # Counting the number of logs the word appears in
    #         if gram not in gram_index_dict:
    #             gram_index_dict[gram] = set()
    #         gram_index_dict[gram].add(index)
    #         # Also the number of different positions
    #         if gram not in gram_index_ilf_dict:
    #             gram_index_ilf_dict[gram] = set()
    #         gram_index_ilf_dict[gram].add(location)

    # idf_dict = {}
    # ilf_dict = {}
    # total_log_num = len(inputVector)
    # # calculating idf and ilf for each gram
    # for index, gram in enumerate(gram_index_dict):
    #     idf_dict[gram] = calculate_inv_freq(total_log_num, len(gram_index_dict[gram]))
    #     ilf_dict[gram] = calculate_inv_freq(max_length, len(gram_index_ilf_dict[gram]))
    
    # tfidf = []
    # # Creating the idf/ilf vector for each log message
    # for index, line in enumerate(inputVector):
    #     cur_tfidf = np.zeros(len(vocabulary))
    #     for gram_index in line:
    #         if opts.add_ilf:
    #             cur_tfidf[gram_index] = (
    #                 float(line.count(gram_index)) * ilf_dict[gram_index] #THIS IS WHERE ILF IS PUT IN IDF PLACE!!!
    #             )
    #         else:
    #             cur_tfidf[gram_index] = (
    #                 float(line.count(gram_index)) * idf_dict[gram_index]
    #             )
    #     tfidf.append(cur_tfidf)

    # tfidf = np.array(tfidf)
    
    # WHY was it decided not to do a normalized tfidf?
    # calculating normalized tfidf
    # tfidf_mean=np.mean(tfidf,axis=0)
    # tfidf_std=np.std(tfidf,axis=0)
    # tfidf=np.true_divide((tfidf-tfidf_mean),tfidf_std)

    # return tfidf, idf_dict, ilf_dict
    return (*calculate_tf_invf_train(inputVector, vocabulary), {})

# Will reimplement into another method as it's same as what the tfidf for train does in its last block
def calculateTfidfForTest(idf_dict, ilf_dict, inputVector, vocabulary):
    tfidf = []
    # calculating tfidf
    for index, l in enumerate(inputVector):
        cur_tfidf = np.zeros(len(vocabulary))
        for gram_index in l:
            if opts.add_ilf:
                cur_tfidf[gram_index] = (
                    float(l.count(gram_index)) * ilf_dict[gram_index]
                )
            else:
                cur_tfidf[gram_index] = (
                    float(l.count(gram_index)) * idf_dict[gram_index]
                )
        tfidf.append(cur_tfidf)
    tfidf = np.array(tfidf)

    # calculating normalized tfidf
    # tfidf=np.true_divide((tfidf-tfidf_mean),tfidf_std)
    return tfidf


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


def get_top_k_SVM_features(clf, feature_names, y_train, target_names):
    # Why print here?...
    # If it has coef_ it's the SVM so we can use the coefficients to visualize
    # the top features and interpret the results
    if hasattr(clf, "coef_"):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:

            """
                There is a bug, because the length of y_train set
                is not equal to the length of target_names
                HOW DO YOU MEAN THERE'S A BUG? <<<<<<<<<<<-------------------------
            """
            print("top 10 keywords per class:")
            print("len(clf.coef_:" + str(len(clf.coef_)))
            print("len(set(y_train)):" + str(len(set(y_train))))
            print(set(y_train))
            
            for i in set(y_train):
                print(i, target_names[i])
                length = min(len(clf.coef_[i]), 10)
                top10 = np.argsort(clf.coef_[i])[-length:]
                print("class name:" + target_names[i])
                for k in feature_names[top10]:
                    print(" " + k)
                print(
                    trim("%s: %s" % (target_names[i], " ".join(feature_names[top10])))
                )

# #############################################################################
# Benchmark classifiers

# This needs further cleaning
# The global variables, it might be better to avoid them
# let's see how it can be done (apparently they are just being used for printing)
# Get metrics or further details on another method
def benchmark(clf, X_train, y_train, X_test, y_test):
    print("_" * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    approach = 'ilf' if opts.add_ilf else 'idf'
    print(f"{approach} train time: {train_time:f}s")

    t0 = time()
    pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, pred)
    test_time = time() - t0

    # This is just to keep the original benchmark method structure
    # This may not necessarily be called here
    global feature_names
    global target_names
    get_top_k_SVM_features(clf, feature_names, y_train, target_names)

    print(f"{approach} test time:  {test_time:f}s")
    print(f"{approach} accuracy:   {score:f}")
    print(f"{approach} macro-f1:" + str(f1_score(y_test, pred, average="macro")))
    print(f"{approach} micro-f1:" + str(f1_score(y_test, pred, average="micro")))
    print(
        metrics.classification_report(
            y_test, pred, target_names=target_names, digits=5
        )
    )

    pred = list(pred)
    iterations = clf.n_iter_
    print('Iterations: ', iterations)

    return clf, score, train_time, test_time, pred, iterations

# May keep the workaround -  in any case checkout how we did the lottery ticket one
# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()

# USAR EL IF MAIN 
# #############################################################################
if __name__ == "__main__":
    t_start = time()

    X_data = []
    label_dict = {}
    y_data = []
    k_of_kflod = opts.k_of_kflod

    target_names = []
    with open(opts.logs) as IN:
        for line in IN:
            l = line.strip().split()
            label = l[0]
            # ignore INFO logs, only classify anomalous logs
            if label == "unlabeled":
                continue
            if label not in label_dict:
                label_dict[label] = len(label_dict)
                target_names.append(label)
            X_data.append(" ".join(l[2:])) # WHY 2? It's disregarding the second token
            y_data.append(label_dict[label])
    X_data = np.array(X_data)
    y_data = np.array(y_data)

    y_test = []
    y_train = []
    X_test = []
    X_train = []

    # KFold
    skf = ""
    if opts.add_ilf:
        skf = StratifiedKFold(n_splits=k_of_kflod)
    else:
        skf = KFold(n_splits=k_of_kflod)
    skf.get_n_splits(X_data, y_data)
    

    total_iter = 0
    total_train_time = 0
    total_test_time = 0

    clf_names_list = []
    training_time_list = []
    test_time_list = []
    pred_list = []
    y_list = []
    x_save_list = []
    cur_num = 1
    for train_index, test_index in skf.split(X_data, y_data):
        print("\ncur_iteration:%d/%d" % (cur_num, k_of_kflod))
        cur_num += 1
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        # right codes
        # WHY DOES IT DO THIS LESS_TRAIN DATA APPROACH?
        # IT JUST CHANGES TEST FOR TRAIN TO ACHIEVE IT??
        if opts.less_train:
            X_train, X_test = X_data[test_index], X_data[train_index]
            y_train, y_test = y_data[test_index], y_data[train_index]
        else:
            X_train, X_test = X_data[train_index], X_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]
        print(" train data size:" + str(X_train.shape[0]))
        print(" test  data size:" + str(X_test.shape[0]))

        # new code#######################
        # vocabulary is a list including words
        t0 = time()
        print(" build_ngram_vocabulary start")
        vocabulary = build_ngram_vocabulary(n_for_gram, X_train)
        print("  build_ngram_vocabulary end, time=" + str(time() - t0) + "s")

        # WHY IS THIS COMMENTED OUT?
        # if opts.add_ilf:
            # X_train,y_train=setTrainDataForILF(X_train,y_train)

        t0 = time()
        print(" log_to_vector for train start")
        X_train_bag_vector, y_train, X_train_save = log_to_vector(
            n_for_gram, X_train, vocabulary, y_train
        )
        print("  log_to_vector for train end, time=" + str(time() - t0) + "s")
        print(" X_train_bag_vector.shape:" + str(X_train_bag_vector.shape))
        t0 = time()
        print(" log_to_vector for test start")

        X_test_bag_vector, y_test, X_test_save = log_to_vector(
            n_for_gram, X_test, vocabulary, y_test
        )
        print("  log_to_vector for test end, time=" + str(time() - t0) + "s")

        t0 = time()
        print(" calculateTfidfForTrain start")
        X_train, invf_dict = calculate_tf_invf_train(X_train_bag_vector, vocabulary)
        print("  calculateTfidfForTrain end, time=" + str(time() - t0) + "s")
        print(" X_train.shape:" + str(X_train.shape))

        t0 = time()
        print(" calculateTfidfForTest start")
        X_test = create_invf_vector(invf_dict, X_test_bag_vector, vocabulary)
        print("  calculateTfidfForTest end, time=" + str(time() - t0) + "s")

        y_list.append(y_test)
        # WHAT'S THE POINT OF THE x_save_list, x_test_save, isn't it the log input already?
        x_save_list.append(X_test_save)
        # add length to feature vector
        if opts.add_length_vector:
            print(" Adding length as feature")
            X_train = addLengthInFeature(X_train, X_train_bag_vector)
            X_test = addLengthInFeature(X_test, X_test_bag_vector)
            print("  X_train.shape after add lengeth feature:" + str(X_train.shape))

        # print("X_train n_samples: %d, n_features: %d" % (X_train.shape)
        # print("X_test  n_samples: %d, n_features: %d" % X_test.shape)

        # WHY HAVING A SEPARATE NAME??
        # WHAT'S THE POINT OF feature_names vs vocabulary
        feature_names = vocabulary

        # print(X_train)
        # print(X_test)

        # print("Extracting features from the training data using a sparse vectorizer")
        # t0 = time()
        # vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
        #                                  stop_words='english')
        # X_train = vectorizer.fit_transform(X_train)
        # duration = time() - t0
        # print("done in %fs" % (duration))
        # print("n_samples: %d, n_features: %d" % X_train.shape)
        # print()

        # print("Extracting features from the test data using the same vectorizer")
        # t0 = time()
        # X_test = vectorizer.transform(X_test)
        # duration = time() - t0
        # print("done in %fs " % (duration))
        # #print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
        # print("n_samples: %d, n_features: %d" % X_test.shape)
        # print()

        # # mapping from integer feature name to original token string

        # feature_names = vectorizer.get_feature_names()
        # # print(type(feature_names))
        # if opts.select_chi2:
        #     print("Extracting %d best features by a chi-squared test" %
        #           opts.select_chi2)
        #     t0 = time()
        #     ch2 = SelectKBest(chi2, k=opts.select_chi2)
        #     X_train = ch2.fit_transform(X_train, y_train)
        #     X_test = ch2.transform(X_test)
        #     if feature_names:
        #         # keep selected feature names
        #         feature_names = [feature_names[i] for i
        #                          in ch2.get_support(indices=True)]
        #     print("done in %fs" % (time() - t0))
        #     print()

        if feature_names:
            feature_names = np.asarray(feature_names)

        results = []
        """
        for clf, name in (
            (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
            (Perceptron(n_iter=50), "Perceptron"),
            (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
                #(KNeighborsClassifier(n_neighbors=10), "kNN"),
            (RandomForestClassifier(n_estimators=100), "Random forest")):
        print('=' * 80)
        print(name)
        results.append(benchmark(clf))

        for penalty in ["l2", "l1"]:
        print('=' * 80)
        print("%s penalty" % penalty.upper())
        # Train Liblinear model
        results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                            tol=1e-3)))

        # Train SGD model
        results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                                penalty=penalty)))

        # Train SGD with Elastic Net penalty
        print('=' * 80)
        print("Elastic-Net penalty")
        results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                            penalty="elasticnet")))

        # Train NearestCentroid without threshold
        print('=' * 80)
        print("NearestCentroid (aka Rocchio classifier)")
        results.append(benchmark(NearestCentroid()))

        # Train sparse Naive Bayes classifiers
        print('=' * 80)
        print("Naive Bayes")
        results.append(benchmark(MultinomialNB(alpha=.01)))
        results.append(benchmark(BernoulliNB(alpha=.01)))

        """

        print("=" * 80)
        print("LinearSVC  l2 penalty")
        # Train Liblinear model
        clf, score, train_time, test_time, pred, iterations = benchmark(
                LinearSVC(
                    penalty="l2",
                    dual=False,
                    # max_iter=1,
                    tol=total_tol,
                ),
                X_train,
                y_train,
                X_test,
                y_test,
            ) # 1e-3
        
        total_iter += iterations
        total_train_time += train_time
        total_test_time += test_time

        clf_descr = str(clf).split('(')[0]
        results.append(
            (clf_descr, score, train_time, test_time, pred)
        )  
        print("=" * 80)

        results = [[x[i] for x in results] for i in range(5)]
        clf_names, score, training_time, test_time, pred = results
        # The first iteration
        if clf_names_list == []:
            clf_names_list = clf_names
            training_time_list = training_time
            test_time_list = test_time
            pred_list = pred
        else:
            for i, k in enumerate(training_time):
                training_time_list[i] += k
                test_time_list[i] += test_time[i]
                pred_list[i].extend(pred[i])

    # Aggregates y results for later report
    total_y = []
    for k in y_list:
        total_y.extend(k)
    # This would eventually be saved but I don't see the reason why yet
    total_x_save = []
    for k in x_save_list:
        total_x_save.extend(k)

    # if opts.add_length_vector:
    #     lable_result_log_filename='./result/'+str(k_of_kflod)+'_'+str(n_for_gram)+'_add_'+lable_result_log_filename
    # else:
    #     lable_result_log_filename='./result/'+str(k_of_kflod)+'_'+str(n_for_gram)+'_'+lable_result_log_filename
    # f_yyx = open(lable_result_log_filename,'w')

    if True:
        # if opts.print_report:
        for i, n in enumerate(clf_names_list):
            print("=" * 80)
            print("%s classification report:" % (n))
            print("-" * 80)
            # WHAT IS THIS total_y COMPARING WITH pred_list[i] ???
            # It looks like it could be the predictions from each model from the cross-validation?
            print(
                metrics.classification_report(
                    total_y, pred_list[i], target_names=target_names, digits=5
                )
            )
            print("macro-f1:" + str(f1_score(total_y, pred_list[i], average="macro")))
            print("micro-f1:" + str(f1_score(total_y, pred_list[i], average="micro")))
            print("=" * 80)

    # if opts.print_cm:
    if True:
        for i, n in enumerate(clf_names_list):
            print("=" * 80)
            print("%s confusion matrix:" % (n))
            print("-" * 80)
            print(metrics.confusion_matrix(total_y, pred_list[i]))
            print("=" * 80)

    score = []
    # print accuracy
    print("=" * 80)
    for i, n in enumerate(clf_names_list):
        # print('=' * 80)
        cur_score = metrics.accuracy_score(total_y, pred_list[i])
        score.append(cur_score)
        print("%s accuracy:   %f" % (n, cur_score))
    print("=" * 80)

    # WHAT'S THIS FOR?
    for i, k in enumerate(total_y):
        pred = ""
        for j in range(len(pred_list)):
            pred += " " + str(pred_list[j][i])
        # f_yyx.writelines(target_names[int(k)]+" "+str(k)+' '+pred+' '+total_x_save[i]+'\n')

    """
    # make some plots

    indices = np.arange(len(clf_names_list))
    training_time = np.array(training_time_list) / np.max(training_time_list)
    test_time = np.array(test_time_list) / np.max(test_time_list)

    # print(len(indices))
    # print(len(score))
    # print(len(training_time_list))

    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices, score, .2, label="score", color='navy')
    plt.barh(indices + .3, training_time, .2, label="training time",
            color='c')
    plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)
    plt.savefig(fig_path)
    """
    print("iters:" + str(total_iter / k_of_kflod))
    print("training time:" + str(total_train_time))
    print("testing  time:" + str(total_test_time))
    print("total time:" + str((time() - t_start) / 60) + "mins,end")

    # plt.show()

    print("k_of_kflod:" + str(k_of_kflod))
    # print("n_for_gram:"+str(n_for_gram))
    # print('end all')
