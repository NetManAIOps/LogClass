from .utils import (
    load_params,
    file_handling,
    print_params,
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


def init_args():
    """Init command line args used for configuration."""

    parser = init_main_args()
    return parser.parse_args()


def parse_args(args):
    """Parse provided args for runtime configuration."""
    params = parse_main_args(args)
    params.update({'train': False})
    return params


def inference(params, x_data, y_data, target_names):
    # Inference
    # Feature engineering
    x_test, _ = extract_features(x_data, params)
    # Binary training features
    y_test = binary_train_gtruth(y_data)
    # Binary PU estimator with RF
    # Load Trained PU Estimator
    binary_clf_getter =\
        binary_classifier_registry.get_binary_model(
            params['binary_classifier'])
    binary_clf = binary_clf_getter(params)
    binary_clf.load()
    # Anomaly detection
    y_pred_pu = binary_clf.predict(x_test)
    get_accuracy = black_box_report_registry.get_bb_report('acc')
    binary_acc = get_accuracy(y_test, y_pred_pu)

    print(binary_acc)
    for report in params['report']:
        try:
            get_bb_report = black_box_report_registry.get_bb_report(report)
            result = get_bb_report(y_test, y_pred_pu)
        except Exception:
            pass
        else:
            print(f'Binary classification {report} report:')
            print(result)


def main():
    # Init params
    params = parse_args(init_args())
    load_params(params)
    print_params(params)
    file_handling(params)
    # Filter params from raw logs
    if "raw_logs" in params:
        preprocess = preprocess_registry.get_preprocessor(params['logs_type'])
        preprocess(params)
    # Load filtered params from file
    print('Loading logs')
    x_data, y_data, target_names = load_logs(params)
    inference(params, x_data, y_data, target_names)


if __name__ == "__main__":
    main()
