from sklearn.model_selection import StratifiedKFold
from .utils import (
    save_params,
    load_params,
    file_handling,
    TestingParameters,
    print_params,
)
from .preprocess import registry as preprocess_registry
from .preprocess.utils import load_logs
from .feature_engineering.utils import (
    binary_train_gtruth,
    multi_features,
    extract_features,
)
from tqdm import tqdm
from .models import binary_registry as binary_classifier_registry
from .models import multi_registry as multi_classifier_registry
from .reporting import bb_registry as black_box_report_registry
from .reporting import wb_registry as white_box_report_registry
from .init_params import init_main_args, parse_main_args


def init_args():
    """Init command line args used for configuration."""

    parser = init_main_args()
    return parser.parse_args()


def parse_args(args):
    """Parse provided args for runtime configuration."""
    params = parse_main_args(args)
    return params


def inference(params, x_data, y_data, target_names):
    # Inference
    # Feature engineering
    x_test, vocabulary = extract_features(x_data, params)
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
    # MultiClass remove healthy logs
    x_infer_multi, y_infer_multi = multi_features(x_test, y_data)
    # Load MultiClass
    multi_classifier_getter =\
        multi_classifier_registry.get_multi_model(params['multi_classifier'])
    multi_classifier = multi_classifier_getter(params)
    multi_classifier.load()
    # Anomaly Classification
    pred = multi_classifier.predict(x_infer_multi)
    get_multi_acc = black_box_report_registry.get_bb_report('multi_acc')
    score = get_multi_acc(y_infer_multi, pred)

    print(binary_acc, score)
    for report in params['report']:
        try:
            get_bb_report = black_box_report_registry.get_bb_report(report)
            result = get_bb_report(y_test, y_pred_pu)
        except Exception:
            pass
        else:
            print(f'Binary classification {report} report:')
            print(result)

        try:
            get_bb_report = black_box_report_registry.get_bb_report(report)
            result = get_bb_report(y_infer_multi, pred)
        except Exception:
            pass
        else:
            print(f'Multi classification {report} report:')
            print(result)

        try:
            get_wb_report = white_box_report_registry.get_wb_report(report)
            result =\
                get_wb_report(params, binary_clf.model, vocabulary,
                              target_names=target_names, top_features=5)
        except Exception:
            pass
        else:
            print(f'Multi classification {report} report:')
            print(result)


def train(params, x_data, y_data, target_names):
    # KFold Cross Validation
    kfold = StratifiedKFold(n_splits=params['kfold']).split(x_data, y_data)
    best_pu_fs = 0.
    best_multi = 0.
    for train_index, test_index in tqdm(kfold):
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        x_train, vocabulary = extract_features(x_train, params)
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
        # Multi-class training features
        x_train_multi, y_train_multi =\
            multi_features(x_train, y_train)
        x_test_multi, y_test_multi = multi_features(x_test, y_test)
        # MultiClass
        multi_classifier_getter =\
            multi_classifier_registry.get_multi_model(params['multi_classifier'])
        multi_classifier = multi_classifier_getter(params)
        multi_classifier.fit(x_train_multi, y_train_multi)
        pred = multi_classifier.predict(x_test_multi)
        get_multi_acc = black_box_report_registry.get_bb_report('multi_acc')
        score = get_multi_acc(y_test_multi, pred)
        better_results = (
            binary_acc > best_pu_fs
            or (binary_acc == best_pu_fs and score > best_multi)
        )

        if better_results:
            if binary_acc > best_pu_fs:
                best_pu_fs = binary_acc
            save_params(params)
            if score > best_multi:
                best_multi = score
            print(binary_acc, score)

        # TryCatch are necessary since I'm trying to consider all 
        # reports the same when they are not 
        for report in params['report']:
            try:
                get_bb_report = black_box_report_registry.get_bb_report(report)
                result = get_bb_report(y_test_pu, y_pred_pu)
            except Exception:
                pass
            else:
                print(f'Binary classification {report} report:')
                print(result)

            try:
                get_bb_report = black_box_report_registry.get_bb_report(report)
                result = get_bb_report(y_test_multi, pred)
            except Exception:
                pass
            else:
                print(f'Multi classification {report} report:')
                print(result)

            try:
                get_wb_report = white_box_report_registry.get_wb_report(report)
                result =\
                    get_wb_report(params, multi_classifier.model, vocabulary,
                                  target_names=target_names, top_features=5)
            except Exception:
                pass
            else:
                print(f'Multi classification {report} report:')
                print(result)


def main():
    # Init params
    params = parse_args(init_args())
    if not params['train']:
        load_params(params)
    print_params(params)
    file_handling(params)  # TODO: handle the case when the experiment ID already exists - this I think is the only one that matters
    # Filter params from raw logs
    if 'raw_logs' in params:
        preprocess = preprocess_registry.get_preprocessor(params['logs_type'])
        preprocess(params)
    # Load filtered params from file
    x_data, y_data, target_names = load_logs(params)
    if params['train']:
        train(params, x_data, y_data, target_names)
    else:
        inference(params, x_data, y_data, target_names)


if __name__ == "__main__":
    main()
