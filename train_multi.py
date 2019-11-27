from sklearn.model_selection import StratifiedKFold
from .utils import (
    save_params,
    file_handling,
    TestingParameters,
    print_params,
    save_results,
)
from .preprocess import registry as preprocess_registry
from .preprocess.utils import load_logs
from .feature_engineering.utils import (
    multi_features,
    extract_features,
)
from tqdm import tqdm
from .models import multi_registry as multi_classifier_registry
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


def init_results():
    results = {
        'exp_name': [],
        'logs_type': [],
        'macro': [],
        'micro': [],
        'train_time': [],
        'run_time': [],
    }
    return results


def add_result(results, params, macro, micro, train_time, run_time):
    results['exp_name'].append(params['id'])
    results['logs_type'].append(params['logs_type'])
    results['macro'].append(macro)
    results['micro'].append(micro)
    results['train_time'].append(train_time)
    results['run_time'].append(run_time)


def train(params, x_data, y_data, target_names):
    results = init_results()
    # KFold Cross Validation
    kfold = StratifiedKFold(n_splits=params['kfold']).split(x_data, y_data)
    best_multi = 0.
    for train_index, test_index in tqdm(kfold):
        # Test & Train are interchanged to enable testing with 10% of the data
        if params['swap']:
            x_test, x_train = x_data[train_index], x_data[test_index]
            y_test, y_train = y_data[train_index], y_data[test_index]
        else:
            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]
        x_train, _ = extract_features(x_train, params)
        print(y_train.shape, y_test.shape)
        with TestingParameters(params):
            x_test, _ = extract_features(x_test, params)
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
        get_multi_acc = black_box_report_registry.get_bb_report('macro')
        macro = get_multi_acc(y_test_multi, pred)
        get_multi_acc = black_box_report_registry.get_bb_report('micro')
        micro = get_multi_acc(y_test_multi, pred)
        better_results = macro > best_multi
        if better_results:
            save_params(params)
            best_multi = macro
            print(macro)

        add_result(
            results,
            params,
            macro,
            micro,
            multi_classifier.train_time,
            multi_classifier.run_time
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
    x_data, y_data, target_names = load_logs(params)
    train(params, x_data, y_data, target_names)


if __name__ == "__main__":
    main()
