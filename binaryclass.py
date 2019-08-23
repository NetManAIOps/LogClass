#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function
import logging
import numpy as np
from time import time
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from .puLearning.puAdapter import PUAdapter
from sklearn.metrics import precision_recall_fscore_support
from .vectorizer import (
    build_ngram_vocabulary,
    log_to_vector,
    calculate_tf_invf_train,
    create_invf_vector,
)
from .utils import addLengthInFeature
import argparse
import matplotlib

matplotlib.use("Agg")


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


def init_flags():
    """Init command line flags used for configuration."""

    parser = argparse.ArgumentParser(
        description="Runs binary classification with "
                    + "PULearning to detect anomalous logs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--logs",
        metavar="logs",
        type=str,
        nargs=1,
        default="./LogClass/data/logs_without_paras.txt",
        help="input logs file path",
    )
    parser.add_argument(
        "--kfold",
        metavar="kfold",
        type=int,
        nargs=1,
        default=3,
        help="kfold crossvalidation",
    )
    parser.add_argument(
        "--iterations",
        metavar="iterations",
        type=int,
        nargs=1,
        default=10,
        help="number of training iterations",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        nargs=1,
        default="unlabeled",
        help="the labels of unlabeled logs",
    )
    parser.add_argument(
        "--add_ilf",
        type=int,
        nargs=1,
        default=1,
        help="if set 1, LogClass will use ilf to generate ferture vector",
    )
    parser.add_argument(
        "--add_length",
        action="store_true",
        default=True,
        help="if set, LogClass will add length as feature",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        default=False,
        help="Print a detailed classification report.",
    )
    parser.add_argument(
        "--top10",
        action="store_true",
        default=False,
        help="Print ten most discriminative terms"
        + " per class for every classifier.",
    )

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
    }

    print("{:-^80}".format("params"))
    print("Beginning binary classification "
          + "using the following configuration:\n")
    for param, value in params.items():
        print("\t{:>13}: {}".format(param, value))
    print()
    print("-" * 80)
    return params


def testPU(X_train, y_train, X_test, y_test, total_pred_y_pu, total_pred_y_re):
    np.random.seed(5)
    # pu_pred=[]
    permut = np.random.permutation(len(y_train))
    X_train = X_train[permut]
    y_train = y_train[permut]

    print("Training set contains ", len(y_train), " examples")
    print(len(np.where(y_train == -1.0)[0]), " are bening(unlabled)")
    print(len(np.where(y_train == 1.0)[0]), " are malignant")
    print()
    isEmpty = False
    pu_f1_scores = []
    reg_f1_scores = []
    step = len(np.where(y_train == 1.0)[0]) / (
        pu_iter_time * multiple_for_pu_iter_time + 1
    )
    n_sacrifice_iter = []
    for i in range(pu_iter_time):
        n_sacrifice_iter.append(i * step)
    print("n_sacrifice_iter", n_sacrifice_iter)
    if total_pred_y_pu == []:
        isEmpty = True
    for i, n_sacrifice in enumerate(n_sacrifice_iter):
        print("=" * 80)
        # send some positives to the negative class! :)
        print("PU transformation in progress.")
        print("Making ", n_sacrifice, " malignant examples bening.")
        print()
        y_train_pu = np.copy(y_train)
        pos = np.where(y_train == 1.0)[0]
        np.random.shuffle(pos)
        sacrifice = pos[: int(n_sacrifice)]
        y_train_pu[sacrifice] = -1.0

        print("PU transformation applied. We now have:")
        print(len(np.where(y_train_pu == -1.0)[0]), " are bening")
        print(len(np.where(y_train_pu == 1.0)[0]), " are malignant")
        print("-" * 80)

        # Get f1 score with pu_learning
        print("PU learning in progress...")
        estimator = RandomForestClassifier(
            n_estimators=10,  # #_of_submodel
            criterion="entropy",  # gini or entropy(default=”gini”)是计算属性的gini(基尼不纯度)还是entropy(信息增益)，来选择最合适的节点。
            bootstrap=True,  # bootstrap=True：是否有放回的采样。
            n_jobs=1,
        )  # 并行job个数   1=不并行；n：n个并行；-1：CPU有多少core，就启动多少job
        pu_estimator = PUAdapter(estimator)
        pu_estimator.fit(X_train, y_train_pu)
        y_pred = pu_estimator.predict(X_test)
        if isEmpty:
            total_pred_y_pu.append(list(y_pred))
        else:
            total_pred_y_pu[i].extend(list(y_pred))
        precision, recall, f1_score, _ =\
            precision_recall_fscore_support(y_test, y_pred)
        pu_f1_scores.append(f1_score[1])
        print("F1 score: ", f1_score[1])
        print("Precision: ", precision[1])
        print("Recall: ", recall[1])
        print()

        # Get f1 score without pu_learning
        print("Regular learning in progress...")
        estimator = RandomForestClassifier(n_estimators=10,
                                           bootstrap=True, n_jobs=1)
        estimator.fit(X_train, y_train_pu)
        y_pred = estimator.predict(X_test)
        if isEmpty:
            total_pred_y_re.append(list(y_pred))
        else:
            total_pred_y_re[i].extend(list(y_pred))
        precision, recall, f1_score, _ =\
            precision_recall_fscore_support(y_test, y_pred)
        reg_f1_scores.append(f1_score[1])
        print("F1 score: ", f1_score[1])
        print("Precision: ", precision[1])
        print("Recall: ", recall[1])

        print("=" * 80)
        print("\n")

    # plt.title("Random forest with/without PU learning")
    # plt.plot(n_sacrifice_iter, pu_f1_scores, label='PU Adapted Random Forest')
    # plt.plot(n_sacrifice_iter, reg_f1_scores, label='Random Forest')
    # plt.xlabel('Number of positive examples hidden in the unlabled set')
    # plt.ylabel('F1 Score')
    # plt.legend()
    # plt.show()

    # return total_pred_y_pu


# #############################################################################
# Benchmark classifiers
def benchmark(clf, X_train, y_train, X_test, y_test):
    print("_" * 80)
    print("LinearSVC Training: ")
    # print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, "coef_"):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

    # if opts.print_report:
    #     print("classification report:")
    #     print(metrics.classification_report(y_test, pred,
    #                                         target_names=target_names))

    # if opts.print_cm:
    #     print("confusion matrix:")
    #     print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split("(")[0]
    pred = list(pred)
    # print(pred)

    precision, recall, f1_score, _ =\
        precision_recall_fscore_support(y_test, pred)
    print("F1 score: ", f1_score)
    print("Precision: ", precision)
    print("Recall: ", recall)

    return clf_descr, score, train_time, test_time, pred
    # results = [[x[i] for x in results] for i in range(5)]


if __name__ == "__main__":
    # Init hparams
    params = parse_args(init_flags())

    input_path = params["logs"]
    k_of_kflod = params["kfold"]
    pu_iter_time = params["iterations"]  # iters of pu test
    unlabel_label = params["prefix"]

    n_for_gram = 1
    multiple_for_pu_iter_time = (
        2
    )

    t_start = time()
    X_data = []
    label_dict = {}
    y_data = []
    target_names = []
    # THIS IS ONLY FOR THE CURRENT KIND OF DATA
    # IT SHOULD BE INDEPENDANT OF THE DATA SOURCE
    with open(input_path) as IN:
        for line in IN:
            L = line.strip().split()
            label = L[0]
            if label != unlabel_label:
                label = "Anomaly"

            if label not in label_dict:
                if label == unlabel_label:
                    label_dict[label] = -1.0
                else:
                    label_dict[label] = 1.0
                target_names.append(label)
            X_data.append(" ".join(L[2:]))
            y_data.append(label_dict[label])
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    # print(set(y_data))
    y_test = []
    y_train = []
    X_test = []
    X_train = []

    # KFold
    skf = StratifiedKFold(n_splits=k_of_kflod)
    skf.get_n_splits(X_data, y_data)
    cur_num = 1

    clf_names_list = []
    training_time_list = []
    test_time_list = []
    pred_list = []
    y_list = []
    x_save_list = []

    total_pred_y_pu = []
    total_pred_y_re = []
    for train_index, test_index in skf.split(X_data, y_data):
        print("=" * 80)
        print("\ncur_iteration:%d/%d" % (cur_num, k_of_kflod))
        cur_num += 1
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        print(" train data size:" + str(X_train.shape[0]))
        print(" test  data size:" + str(X_test.shape[0]))

        # new code#######################
        # vocabulary is a list including words
        t0 = time()
        print(" ngramDivide start")
        vocabulary = build_ngram_vocabulary(n_for_gram, X_train)
        print("  ngramDivide end, time=" + str(time() - t0) + "s")

        t0 = time()
        print(" convertLogToVector for train start")
        X_train_bag_vector, y_train = log_to_vector(
            n_for_gram, X_train, vocabulary, y_train
        )
        print("  convertLogToVector for train end, time="
              + str(time() - t0) + "s")
        print(" X_train_bag_vector.shape:" + str(X_train_bag_vector.shape))
        t0 = time()
        print(" convertLogToVector for test start")

        X_test_bag_vector, y_test = log_to_vector(
            n_for_gram, X_test, vocabulary, y_test
        )
        print("  convertLogToVector for test end, time="
              + str(time() - t0) + "s")

        t0 = time()
        print(" calculateTfidfForTrain start")
        X_train, invf_dict = calculate_tf_invf_train(X_train_bag_vector,
                                                     vocabulary)
        print("  calculateTfidfForTrain end, time=" + str(time() - t0) + "s")
        print(" X_train.shape:" + str(X_train.shape))

        t0 = time()
        print(" calculateTfidfForTest start")
        X_test = create_invf_vector(invf_dict, X_test_bag_vector, vocabulary)
        print("  calculateTfidfForTest end, time=" + str(time() - t0) + "s")
        y_test = np.array(y_test)
        y_train = np.array(y_train)
        y_list.append(y_test)
        if params["add_length"]:
            print(" Adding length as feature")
            X_train = addLengthInFeature(X_train, X_train_bag_vector)
            X_test = addLengthInFeature(X_test, X_test_bag_vector)
            print("  X_train.shape after add lengeth feature:"
                  + str(X_train.shape))

        feature_names = vocabulary

        if feature_names:
            feature_names = np.asarray(feature_names)

        results = []

        print("=" * 80)
        testPU(X_train, y_train, X_test, y_test,
               total_pred_y_pu, total_pred_y_re)
        print("=" * 80)
        results.append(
            benchmark(
                LinearSVC(penalty="l2", dual=False, tol=1e-3),
                X_train,
                y_train,
                X_test,
                y_test,
            )
        )

    # This is used for comparing results. Not sure it's still required
    total_y = []
    for k in y_list:
        total_y.extend(k)

    for i, n in enumerate(total_pred_y_pu):
        # precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)
        print("=" * 80)
        print(str(i) + "/" + str(pu_iter_time))
        # print("puLearning classification report:" )

        # print(metrics.classification_report(total_y, total_pred_y_pu[i],
        #                                     target_names=target_names))
        # print("confusion matrix:" )
        # print(metrics.confusion_matrix(total_y, total_pred_y_pu[i]))
        precision1, recall1, f1_score1, _ = precision_recall_fscore_support(
            total_y, total_pred_y_pu[i]
        )
        print(
            "puLearning classifier: precision:"
            + str(round(precision1[1], 4))
            + " recall:"
            + str(round(recall1[1], 4))
            + " f1_score:"
            + str(round(f1_score1[1], 4))
        )
        print("-" * 80)
        # print(str(i)+'/'+str(pu_iter_time))
        # print("regular classification report:" )
        # print(metrics.classification_report(total_y, total_pred_y_re[i],
        #                                     target_names=target_names))

        # print("confusion matrix:" )
        # print(metrics.confusion_matrix(total_y, total_pred_y_re[i]))
        precision2, recall2, f1_score2, _ = precision_recall_fscore_support(
            total_y, total_pred_y_re[i]
        )
        print(
            "regular classifier: precision:"
            + str(round(precision2[1], 4))
            + " recall:"
            + str(round(recall2[1], 4))
            + " f1_score:"
            + str(round(f1_score2[1], 4))
        )

        print("=" * 80)

    score = []
    # print accuracy
    print("=" * 80)
    for i, n in enumerate(clf_names_list):
        # print('=' * 80)
        cur_score = metrics.accuracy_score(total_y, pred_list[i])
        score.append(cur_score)
        print("%s accuracy:   %0.3f" % (n, cur_score))
    print("=" * 80)

    for i, k in enumerate(total_y):
        pred = ""
        for j in range(len(pred_list)):
            pred += " " + str(pred_list[j][i])

    print("total time: " + str(int((time() - t_start))) + "s")