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
import argparse
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
from .vectorizer import get_lf, build_ngram_vocabulary, log_to_vector, calculate_inv_freq, calculate_tf_invf_train, create_invf_vector
from .utils import trim, addLengthInFeature
import sys

# input_path='./data/logs_without_paras.txt'
# k_of_kflod=10
n_for_gram = 1
total_tol = 1e-1  # param of svc


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def init_flags():
    """Init command line flags used for configuration."""

    parser = argparse.ArgumentParser(
        description="Runs binary classification with PULearning to detect anomalous logs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--logs', metavar="logs", type = str, nargs=1, default = './LogClass/data/logs_without_paras.txt', help = 'input logs file path')
    parser.add_argument('--kfold', metavar="kfold", type = int, nargs=1, default = 3, help = 'kfold crossvalidation')
    parser.add_argument('--iterations', metavar = 'iterations', type = int, nargs=1, default = 10, help="number of training iterations")
    parser.add_argument('--prefix', type = str, nargs=1, default = "unlabeled", help = 'the labels of unlabeled logs')
    parser.add_argument('--add_ilf', type = int, nargs=1, default = False, help = 'if set, LogClass will use ilf to generate ferture vector')
    parser.add_argument('--add_length', action="store_true", default = False, help = 'if set, LogClass will add length as feature')
    parser.add_argument('--report', action="store_true", default = False, help = 'Print a detailed classification report.')
    parser.add_argument('--top10', action="store_true", default = False, help = 'Print ten most discriminative terms per class for every classifier.')
    parser.add_argument('--less_train', action="store_true", default = False, help = 'Use less train data.')

    # May be useful to remember for lists
    # parser.add_argument(
    #     "--experiment",
    #     metavar="experiment",
    #     type=str,
    #     nargs=1,
    #     default=["reinit_orig"],
    #     choices=["no_pruning", "reinit_rand", "reinit_orig", "reinit_none"],
    #     help="the experiment to run",
    # )

    return parser.parse_args()

def parse_args(args):
    """Parse provided args for runtime configuration."""
    params = {
        "logs": args.logs,
        "kfold": args.kfold,
        "iterations": args.iterations,
        "prefix": args.prefix,
        "add_ilf": args.add_ilf,
        "add_length": args.add_length,
        "report": args.report,
        "top10": args.top10,
        "less_train": args.less_train
    }
    
    print("{:-^80}".format("params"))
    print("Beginning binary classification using the following configuration:\n")
    for param, value in params.items():
        print("\t{:>13}: {}".format(param, value))
    print()
    print("-" * 80)
    return params

def get_top_k_SVM_features(clf, feature_names, y_train, target_names):
    # Why print here?...
    # If it has coef_ it's the SVM so we can use the coefficients to visualize
    # the top features and interpret the results
    if hasattr(clf, "coef_"):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if params['top10'] and feature_names is not None:

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
    approach = 'ilf' if params['add_ilf'] else 'idf'
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

# USAR EL IF MAIN 
# #############################################################################
if __name__ == "__main__":
    params = parse_args(init_flags())

    t_start = time()

    X_data = []
    label_dict = {}
    y_data = []
    k_of_kflod = params['kfold']

    target_names = []
    with open(params['logs']) as IN:
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
    if params['add_ilf']:
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
        if params['less_train']:
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
        if params['add_length']:
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
