from sklearn.model_selection import StratifiedKFold
from .utils import (
    save_params,
    file_handling,
    TestingParameters,
    print_params,
)
from .preprocess import registry as preprocess_registry
from .preprocess.utils import load_logs
from .feature_engineering.utils import (
    binary_train_gtruth,
    extract_features,
)
from tqdm import tqdm
from .models import binary_registry as binary_classifier_registry
from .reporting import bb_registry as black_box_report_registry
from .init_params import init_main_args, parse_main_args


def init_args():
    """Init command line args used for configuration."""

    parser = init_main_args()
    return parser.parse_args()


def parse_args(args):
    """Parse provided args for runtime configuration."""
    params = parse_main_args(args)
    params.update({'train': True})
    return params


def train(params, x_data, y_data, target_names):
    # KFold Cross Validation
    kfold = StratifiedKFold(n_splits=params['kfold']).split(x_data, y_data)
    best_pu_fs = 0.
    for train_index, test_index in tqdm(kfold):
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        x_train, _ = extract_features(x_train, params)
        with TestingParameters(params):
            x_test, _ = extract_features(x_test, params)
        # Binary training features
        y_test_pu = binary_train_gtruth(y_test)
        y_train_pu = binary_train_gtruth(y_train)
        # Binary PULearning with RF
        binary_clf_getter =\
            binary_classifier_registry.get_binary_model(
                params['binary_classifier'])
        binary_clf = binary_clf_getter(params)
        binary_clf.fit(x_train, y_train_pu)
        y_pred_pu = binary_clf.predict(x_test)
        get_accuracy = black_box_report_registry.get_bb_report('acc')
        binary_acc = get_accuracy(y_test_pu, y_pred_pu)
        better_results = binary_acc > best_pu_fs
        if better_results:
            if binary_acc > best_pu_fs:
                best_pu_fs = binary_acc
            save_params(params)
            binary_clf.save()
            print(binary_acc)

        for report in params['report']:
            try:
                get_bb_report = black_box_report_registry.get_bb_report(report)
                result = get_bb_report(y_test_pu, y_pred_pu)
            except Exception:
                pass
            else:
                print(f'Binary classification {report} report:')
                print(result)


def main():
    # Init params
    params = parse_args(init_args())
    file_handling(params)
    # Filter params from raw logs
    if "raw_logs" in params:
        preprocess = preprocess_registry.get_preprocessor(params['logs_type'])
        preprocess(params)
    # Load filtered params from file
    print('Loading logs')
    x_data, y_data, target_names = load_logs(params)
    print_params(params)
    train(params, x_data, y_data, target_names)


if __name__ == "__main__":
    main()
