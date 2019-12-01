from sklearn.model_selection import StratifiedKFold
from .utils import (
    file_handling,
    TestingParameters,
    print_params,
    save_results,
)
from .preprocess import registry as preprocess_registry
from .preprocess.utils import load_logs
from .feature_engineering.utils import (
    binary_train_gtruth,
    extract_features,
)
from .models import binary_registry as binary_classifier_registry
from .reporting import bb_registry as black_box_report_registry
from .init_params import init_main_args, parse_main_args
import numpy as np


def init_args():
    """Init command line args used for configuration."""

    parser = init_main_args()
    parser.add_argument(
        "--ratio",
        metavar="ratio",
        type=int,
        nargs=1,
        default=[8],
        help="ratio",
    )
    parser.add_argument(
        "--top_percentage",
        metavar="top_percentage",
        type=int,
        nargs=1,
        default=[11],
        help="top_percentage",
    )
    parser.add_argument(
        "--step",
        metavar="step",
        type=int,
        nargs=1,
        default=[2],
        help="step",
    )
    return parser.parse_args()


def parse_args(args):
    """Parse provided args for runtime configuration."""
    params = parse_main_args(args)
    additional_params = {
                            "ratio": args.ratio[0],
                            "top_percentage": args.top_percentage[0],
                            "step": args.step[0],
                            "train": True,
                        }
    params.update(additional_params)
    return params


def force_ratio(params, x_data, y_data):
    """Force a ratio between anomalous and healthy logs"""
    ratio = params['ratio']
    if ratio > 0:
        anomalous = np.where(y_data == 1.0)[0]
        healthy = np.where(y_data == -1.0)[0]
        if len(anomalous) * ratio <= len(healthy):
            keep_anomalous = len(anomalous)
            keep_healthy = keep_anomalous * ratio
        else:
            keep_anomalous = len(healthy) // ratio
            keep_healthy = len(healthy)
        np.random.seed(10)
        permut = np.random.permutation(len(healthy))
        keep = permut[:keep_healthy]
        healthy = healthy[keep]
        permut = np.random.permutation(len(anomalous))
        keep = permut[:keep_anomalous]
        anomalous = anomalous[keep]
        result = sorted(np.concatenate((anomalous, healthy)))
        y_data = y_data[result]
        x_data = x_data[result]
        return x_data, y_data


def init_results(params):
    results = {
        'exp_name': [],
        'logs_type': [],
        'percentage': [],
        'pu_f1': [],
        f"{params['binary_classifier']}_f1": [],
    }
    return results


def add_result(results, params, percentage, pu_acc, b_clf_acc):
    results['exp_name'].append(params['id'])
    results['logs_type'].append(params['logs_type'])
    results['percentage'].append(percentage)
    results['pu_f1'].append(pu_acc)
    results[f"{params['binary_classifier']}_f1"].append(b_clf_acc)


def run_test(params, x_data, y_data):
    results = init_results(params)
    # Binary training features
    y_data = binary_train_gtruth(y_data)
    x_data, y_data = force_ratio(params, x_data, y_data)
    print("total logs", len(y_data))
    print(len(np.where(y_data == -1.0)[0]), " are unlabeled")
    print(len(np.where(y_data == 1.0)[0]), " are anomalous")
    # KFold Cross Validation
    kfold = StratifiedKFold(n_splits=params['kfold']).split(x_data, y_data)
    for train_index, test_index in kfold:
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        x_train, _ = extract_features(x_train, params)
        with TestingParameters(params):
            x_test, _ = extract_features(x_test, params)
        np.random.seed(5)
        permut = np.random.permutation(len(y_train))
        x_train = x_train[permut]
        y_train = y_train[permut]
        top_percentage = params['top_percentage']
        step = params['step']
        # Relabeling anomalous logs to unlabeled to test PU Learning Robustness
        for i in range(0, top_percentage, step):
            y_train_pu = np.copy(y_train)
            if i > 0:
                n_unlabeled = len(np.where(y_train_pu == -1.0)[0])
                sacrifice_size = (i*n_unlabeled)//(100 - i)
                print(i, n_unlabeled, sacrifice_size)
                pos = np.where(y_train == 1.0)[0]
                np.random.shuffle(pos)
                sacrifice = pos[: sacrifice_size]
                y_train_pu[sacrifice] = -1.0

            print(f"{i}% of anomalous log in unlabeled logs:")
            print(len(np.where(y_train_pu == -1.0)[0]), " are unlabeled")
            print(len(np.where(y_train_pu == 1.0)[0]), " are anomalous")
            # Binary PULearning with RF
            pu_getter =\
                binary_classifier_registry.get_binary_model("pu_learning")
            binary_clf = pu_getter(params)
            binary_clf.fit(x_train, y_train_pu)
            y_pred_pu = binary_clf.predict(x_test)
            get_accuracy = black_box_report_registry.get_bb_report('acc')
            pu_acc = get_accuracy(y_test, y_pred_pu)
            # Comparing given Binary Classifier with PU Learning
            comparison_clf_getter =\
                binary_classifier_registry.get_binary_model(
                    params['binary_classifier'])
            binary_clf = comparison_clf_getter(params)
            binary_clf.fit(x_train, y_train_pu)
            y_pred = binary_clf.predict(x_test)
            b_clf_acc = get_accuracy(y_test, y_pred)
            print(f"PU Acc: {pu_acc}\n{params['binary_classifier']}"
                  + " Acc: {b_clf_acc}")

            add_result(
                results,
                params,
                i,
                pu_acc,
                b_clf_acc
            )

    save_results(results, params)


def main():
    # Init params
    params = parse_args(init_args())
    print_params(params)
    file_handling(params)
    # Filter params from raw logs
    if "raw_logs" in params:
        preprocess = preprocess_registry.get_preprocessor(params['logs_type'])
        preprocess(params)
    # Load filtered params from file
    print('Loading logs')
    x_data, y_data, _ = load_logs(params)
    run_test(params, x_data, y_data)


if __name__ == "__main__":
    main()
