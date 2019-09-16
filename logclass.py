import numpy as np
import argparse
import sys
import os
import shutil
import subprocess
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from .puLearning.puAdapter import PUAdapter
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.externals import joblib
from .vectorizer import (
    get_tf,
    get_lf,
    calculate_idf,
    calculate_ilf,
    build_vocabulary,
    log_to_vector,
    calculate_tf_invf_train,
    create_invf_vector,
)
from .utils import addLengthInFeature
import pickle
import json


def init_flags():
    """Init command line flags used for configuration."""

    parser = argparse.ArgumentParser(
        description="Runs binary classification with "
                    + "PULearning to detect anomalous logs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw_logs",
        metavar="raw_logs",
        type=str,
        nargs=1,
        default=["./LogClass/data/rawlog.txt"],
        help="input logs file path",
    )
    base_dir_default = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "outputs"
    )
    parser.add_argument(
        "--base_dir",
        metavar="base_dir",
        type=str,
        nargs=1,
        default=[base_dir_default],
        help="base output directory for pipeline output files",
    )
    parser.add_argument(
        "--logs",
        metavar="logs",
        type=str,
        nargs=1,
        default=[os.path.join(base_dir_default, "logs_without_paras.txt")],
        help="input logs file path",
    )
    parser.add_argument(
        "--kfold",
        metavar="kfold",
        type=int,
        nargs=1,
        default=[3],
        help="kfold crossvalidation",
    )
    parser.add_argument(
        "--iterations",
        metavar="iterations",
        type=int,
        nargs=1,
        default=[10],
        help="number of training iterations",
    )
    parser.add_argument(
        "--healthy_label",
        type=str,
        nargs=1,
        default=["unlabeled"],
        help="the labels of unlabeled logs",
    )
    parser.add_argument(
        "--add_ilf",
        action="store_true",
        default=True,
        help="if set, LogClass will use ilf to generate ferture vector",
    )
    parser.add_argument(
        "--add_length",
        action="store_true",
        default=False,
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
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="If set, logclass will train on the given data. Otherwise"
             + "it will run inference on it.",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        default=False,
        help="If set, the raw logs parameters will be filtered and a "
             + "new file created with the filtered logs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="force training overwriting previous output.",
    )

    return parser.parse_args()


def parse_args(args):
    """Parse provided args for runtime configuration."""
    params = {
        "logs": args.logs[0],
        "raw_logs": args.raw_logs[0],
        "kfold": args.kfold[0],
        "iterations": args.iterations[0],
        "healthy_label": args.healthy_label[0],
        "add_ilf": args.add_ilf,
        "add_length": args.add_length,
        "report": args.report,
        "top10": args.top10,
        "train": args.train,
        "filter": args.filter,
        "force": args.force,
        "base_dir": args.base_dir[0],
    }

    print("{:-^80}".format("params"))
    print("Beginning binary classification "
          + "using the following configuration:\n")
    for param, value in params.items():
        print("\t{:>13}: {}".format(param, value))
    print()
    print("-" * 80)
    return params


def load_logs(log_path, unlabel_label='unlabeled', ignore_unlabeled=False):
    x_data = []
    y_data = []
    label_dict = {}
    target_names = []
    with open(log_path) as IN:
        for line in IN:
            L = line.strip().split()
            label = L[0]
            if label not in label_dict:
                if ignore_unlabeled and label == unlabel_label:
                    continue
                if label == unlabel_label:
                    label_dict[label] = -1.0
                elif label not in label_dict:
                    label_dict[label] = len(label_dict)
                    target_names.append(label)
            x_data.append(" ".join(L[1:]))
            y_data.append(label_dict[label])
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    return x_data, y_data, target_names


def filter_params(params):
    return_code = subprocess.call([
        sys.executable,
        './LogClass/filterparams.py',
        '--input',
        params['raw_logs'],
        '--output',
        params['logs']
        ])
    if return_code != 0:
        raise RuntimeError(f'filterparams failed, {return_code}')


def calculate_invf_dict(params, train_vector, vocabulary):
    if params['add_ilf']:
        freq = get_lf
        invf = calculate_ilf
    else:
        freq = get_tf
        invf = calculate_idf
    invf_dict = calculate_tf_invf_train(
        train_vector,
        vocabulary,
        get_f=freq,
        calc_invf=invf
        )
    return invf_dict


def get_feature_names(vocabulary, add_length=True):
    feature_names = zip(vocabulary.keys(), vocabulary.values())
    feature_names = sorted(feature_names, key=lambda x: x[1])
    feature_names = [x[0] for x in feature_names]
    if add_length:
        feature_names.append('LENGTH')
    return np.array(feature_names)


def get_top_k_SVM_features(svm_clf: LinearSVC, vocabulary,
                           target_names, top_features=3):
    top_k_label = {}
    feature_names = get_feature_names(vocabulary)
    for i, label in enumerate(target_names):
        coef = svm_clf.coef_[i]
        top_coefficients = np.argsort(coef)[-top_features:]
        top_k_features = feature_names[top_coefficients]
        top_k_label[label] = list(reversed(top_k_features))
    return top_k_label


def main():
    # Init params
    params = parse_args(init_flags())
    if params['train']:
        if os.path.exists(params["base_dir"]) and not params["force"]:
            raise FileExistsError(
                "directory '{} already exists. ".format(params["base_dir"])
                + "Run with --force to overwrite."
            )
        if os.path.exists(params["base_dir"]):
            shutil.rmtree(params["base_dir"])
        os.makedirs(params["base_dir"])
    else:
        if not os.path.exists(params["base_dir"]):
            raise FileNotFoundError(
                "directory '{} doesn't exist. ".format(params["base_dir"])
                + "Run train first before running inference."
            )
    # Filter params from raw logs
    if params['filter']:
        filter_params(params)
    # Load filtered params from file
    x_data, y_data, target_names = load_logs(
        params['logs'],
        unlabel_label=params['healthy_label'])
    if params['train']:
        # KFold Cross Validation
        kfold = StratifiedKFold(n_splits=params['kfold']).split(x_data, y_data)
        best_pu_fs = 0.
        best_multi = 0.
        for train_index, test_index in kfold:
            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]
            # Build Vocabulary
            vocabulary = build_vocabulary(x_train)
            # Feature Engineering
            x_train_vector = log_to_vector(x_train, vocabulary)
            x_test_vector = log_to_vector(x_test, vocabulary)
            invf_dict = calculate_invf_dict(params, x_train_vector, vocabulary)
            x_train = create_invf_vector(x_train_vector, invf_dict, vocabulary)
            x_test = create_invf_vector(x_test_vector, invf_dict, vocabulary)
            # Further feature engineering
            if params["add_length"]:
                x_train = addLengthInFeature(x_train, x_train_vector)
                x_test = addLengthInFeature(x_test, x_test_vector)
            # Binary training features
            y_test_pu = np.where(y_test == -1.0, -1.0, 1.0)
            y_train_pu = np.where(y_train == -1.0, -1.0, 1.0)
            # Binary PULearning with RF
            estimator = RandomForestClassifier(
                n_estimators=10,
                criterion="entropy",
                bootstrap=True,
                n_jobs=-1,
            )
            pu_estimator = PUAdapter(estimator)
            pu_estimator.fit(x_train, y_train_pu)
            y_pred_pu = pu_estimator.predict(x_test)
            pu_precision, pu_recall, pu_f1_score, _ =\
                precision_recall_fscore_support(y_test_pu, y_pred_pu)
            # MultiClass remove healthy logs
            anomalous_train = (y_train != -1)
            anomalous_test = (y_test != -1)
            x_train_multi, y_train_multi =\
                x_train[anomalous_train], y_train[anomalous_train]
            x_test_multi, y_test_multi =\
                x_test[anomalous_test], y_test[anomalous_test]
            # MultiClass
            multi_classifier = LinearSVC(penalty="l2", dual=False, tol=1e-1)
            multi_classifier.fit(x_train_multi, y_train_multi)
            pred = multi_classifier.predict(x_test_multi)
            score = metrics.accuracy_score(y_test_multi, pred)
            if pu_f1_score[1] > best_pu_fs or (pu_f1_score[1] == best_pu_fs and score > best_multi):
                if pu_f1_score[1] > best_pu_fs:
                    best_pu_fs = pu_f1_score[1]
                vocab_file = os.path.join(params['base_dir'], 'vocab.json')
                with open(vocab_file, "w") as fp:
                    json.dump(vocabulary, fp)
                invf_dict_file = os.path.join(params['base_dir'],
                                              'invf_dict.pkl')
                with open(invf_dict_file, "wb") as fp:
                    pickle.dump(invf_dict, fp)
                pu_estimator_file = os.path.join(params['base_dir'],
                                                 'pu_estimator.pkl')
                pu_saver = {'estimator': pu_estimator.estimator,
                            'c': pu_estimator.c}
                with open(pu_estimator_file, 'wb') as pu_estimator_file:
                    pickle.dump(pu_saver, pu_estimator_file)
                if score > best_multi:
                    best_multi = score
                multi_file = os.path.join(params['base_dir'], 'multi.pkl')
                with open(multi_file, 'wb') as multi_clf_file:                    
                    pickle.dump(multi_classifier, multi_clf_file)
                print(pu_f1_score[1], score)
            print(get_top_k_SVM_features(
                multi_classifier, vocabulary, target_names))
    else:
        # Inference
        vocab_file = os.path.join(params['base_dir'], 'vocab.json')
        with open(vocab_file, "r") as fp:
            vocabulary = json.load(fp)
        invf_dict_file = os.path.join(params['base_dir'],
                                      'invf_dict.pkl')
        with open(invf_dict_file, "rb") as fp:
            invf_dict = pickle.load(fp)
        x_vector = log_to_vector(x_data, vocabulary)
        x_test = create_invf_vector(x_vector, invf_dict, vocabulary)
        # Feature engineering
        if params["add_length"]:
            x_test = addLengthInFeature(x_test, x_vector)
        # Binary training features
        y_test = np.where(y_data == -1.0, -1.0, 1.0)
        # Binary PU estimator with RF
        # Load Trained PU Estimator
        pu_estimator_file = os.path.join(params['base_dir'],
                                         'pu_estimator.pkl')
        with open(pu_estimator_file, 'rb') as pu_estimator_file:
            pu_saver = pickle.load(pu_estimator_file)
            estimator = pu_saver['estimator']
            pu_estimator = PUAdapter(estimator)
            pu_estimator.c = pu_saver['c']
            pu_estimator.estimator_fitted = True
        # Anomaly detection
        y_pred_pu = pu_estimator.predict(x_test)
        pu_precision, pu_recall, pu_f1_score, _ =\
            precision_recall_fscore_support(y_test, y_pred_pu)
        # MultiClass remove healthy logs
        anomalous = (y_data != -1)
        x_infer_multi, y_infer_multi =\
            x_test[anomalous], y_data[anomalous]
        # Load MultiClass
        multi_file = os.path.join(params['base_dir'], 'multi.pkl')
        with open(multi_file, 'rb') as multi_clf_file:
            multi_classifier = pickle.load(multi_clf_file)
        # Anomaly Classification
        pred = multi_classifier.predict(x_infer_multi)
        score = metrics.accuracy_score(y_infer_multi, pred)

        print(pu_f1_score[1], score)


if __name__ == "__main__":
    main()
