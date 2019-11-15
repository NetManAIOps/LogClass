import os
import argparse


def init_main_args():
    """Init command line args used for configuration."""

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
        "--logs_type",
        metavar="logs_type",
        type=str,
        nargs=1,
        default=["original"],
        choices=["original", "bgl", "bgl_old"],
        help="Input type of logs.",
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
        "--healthy_label",
        type=str,
        nargs=1,
        default=["unlabeled"],
        help="the labels of unlabeled logs",
    )
    parser.add_argument(
        "--features",
        metavar="features",
        type=str,
        nargs='+',
        default=["tfilf"],
        choices=["tfidf", "tfilf", "length", "tf"],
        help="Features to be extracted from the logs messages.",
    )
    parser.add_argument(
        "--report",
        metavar="report",
        type=str,
        nargs='+',
        default=["confusion_matrix"],
        choices=["confusion_matrix",
                 "acc",
                 "multi_acc",
                 "top_k_svm",
                 "micro",
                 "macro"
                 ],
        help="Reports to be generated from the model and its predictions.",
    )
    parser.add_argument(
        "--binary_classifier",
        metavar="binary_classifier",
        type=str,
        nargs=1,
        default=["pu_learning"],
        choices=["pu_learning", "regular"],
        help="Binary classifier to be used as anomaly detector.",
    )
    parser.add_argument(
        "--multi_classifier",
        metavar="multi_classifier",
        type=str,
        nargs=1,
        default=["svm"],
        choices=["svm"],
        help="Multi-clas classifier to classify anomalies.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="If set, logclass will train on the given data. Otherwise"
             + "it will run inference on it.",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        default=False,
        help="If set, the raw logs parameters will be preprocessed and a "
             + "new file created with the preprocessed logs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="force training overwriting previous output.",
    )

    return parser


def parse_main_args(args):
    """Parse provided args for runtime configuration."""
    params = {
        "logs": args.logs[0],
        "raw_logs": args.raw_logs[0],
        "kfold": args.kfold[0],
        "healthy_label": args.healthy_label[0],
        "report": args.report,
        "train": args.train,
        "preprocess": args.preprocess,
        "force": args.force,
        "base_dir": args.base_dir[0],
        "logs_type": args.logs_type[0],
        "features": args.features,
        "binary_classifier": args.binary_classifier[0],
        "multi_classifier": args.multi_classifier[0],
    }
    return params
