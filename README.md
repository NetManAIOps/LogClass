## LogClass
This repository provides an open-source toolkit for LogClass framework from Meng, Weibin, et al. "[Device-agnostic log anomaly classification with partial labels](https://ieeexplore.ieee.org/abstract/document/8624141)." *2018 IEEE/ACM 26th International Symposium on Quality of Service (IWQoS)*. IEEE, 2018. . 

LogClass combines a word representation method and the [PU learning model](https://github.com/aldro61/pu-learning) to construct a device-agnostic vocabulary based on partial labels. A novel Term Frequency Inverse Location Frequency (TF-ILF) method to properly weight the words of device logs in feature construction was also proposed improving accuracy (measured by Macro-F1) from 93.31% to 99.57% compared to a state-of-the-art method. 

### Table of Contents

[TOC]

### Requirements



### Quick Start

Several experiments using LogClass are included. To run training of the global experiment doing anomaly detection and classification run the following command in the home directory of this project: 

```
python -m LogClass.logclass --train --logs_type "bgl" --raw_logs "./Data/RAS_LOGS" --report macro
```



#### Directory Structure

```
.
├── data
│   ├── bgl_logs_without_paras.txt
│   └── rawlog.txt
├── decorators.py
├── feature_engineering
│   ├── __init__.py
│   ├── length.py
│   ├── registry.py
│   ├── tf_idf.py
│   ├── tf_ilf.py
│   ├── tf.py
│   ├── utils.py
│   └── vectorizer.py
├── init_params.py
├── __init__.py
├── LogClass.pdf
├── logclass.py
├── models
│   ├── __init__.py
│   ├── base_model.py
│   ├── binary_registry.py
│   ├── multi_registry.py
│   ├── pu_learning.py
│   ├── regular.py
│   └── svm.py
├── output
│   ├── preprocessed_logs
│   │   ├── bgl_old.txt
│   │   ├── open_Apache.txt
│   │   └── open_bgl.txt
│   └── train_multi_open_bgl_3312211826
│       ├── best_params.json
│       ├── features
│       │   ├── tfidf.pkl
│       │   └── vocab.pkl
│       ├── models
│       │   └── multi.pkl
│       └── results.csv
├── preprocess
│   ├── __init__.py
│   ├── bgl_paper_data.py
│   ├── bgl_preprocessor.py
│   ├── open_source_logs.py
│   ├── original_preprocessor.py
│   ├── registry.py
│   └── utils.py
├── puLearning
│   ├── __init__.py
│   └── puAdapter.py
├── README.md
├── reporting
│   ├── __init__.py
│   ├── accuracy.py
│   ├── bb_registry.py
│   ├── confusion_matrix.py
│   ├── macrof1.py
│   ├── microf1.py
│   ├── multi_class_acc.py
│   ├── top_k_svm.py
│   └── wb_registry.py
├── run_binary.py
├── test_pu.py
├── train_binary.py
├── train_multi.py
└── utils.py
```

#### Datasets

High overview of the datasets. Maybe cite the sources. See how others show this in their repos.

BGL, Paper BGL, Open-source datasets

#### Run LogParse

##### Arguments

```
python -m LogClass.logclass --help
usage: logclass.py [-h] [--raw_logs raw_logs] [--base_dir base_dir]
                   [--logs logs] [--models_dir models_dir]
                   [--features_dir features_dir] [--logs_type logs_type]
                   [--kfold kfold] [--healthy_label HEALTHY_LABEL]
                   [--features features [features ...]]
                   [--report report [report ...]]
                   [--binary_classifier binary_classifier]
                   [--multi_classifier multi_classifier] [--train]
                   [--preprocess] [--force] [--id id] [--swap]

Runs binary classification with PULearning to detect anomalous logs.

optional arguments:
  -h, --help            show this help message and exit
  --raw_logs raw_logs   input logs file path (default:
                        ['./LogClass/data/rawlog.txt'])
  --base_dir base_dir   base output directory for pipeline output files
                        (default: ['D:\\Federico\\Tsinghua
                        MAC\\NetMan\\LogClass\\LogClass\\output'])
  --logs logs           input logs file path and output for raw logs
                        preprocessing (default: None)
  --models_dir models_dir
                        trained models input/output directory path (default:
                        None)
  --features_dir features_dir
                        trained features_dir input/output directory path
                        (default: None)
  --logs_type logs_type
                        Input type of logs. (default: ['original'])
  --kfold kfold         kfold crossvalidation (default: [3])
  --healthy_label HEALTHY_LABEL
                        the labels of unlabeled logs (default: ['unlabeled'])
  --features features [features ...]
                        Features to be extracted from the logs messages.
                        (default: ['tfilf'])
  --report report [report ...]
                        Reports to be generated from the model and its
                        predictions. (default: ['confusion_matrix'])
  --binary_classifier binary_classifier
                        Binary classifier to be used as anomaly detector.
                        (default: ['pu_learning'])
  --multi_classifier multi_classifier
                        Multi-clas classifier to classify anomalies. (default:
                        ['svm'])
  --train               If set, logclass will train on the given data.
                        Otherwiseit will run inference on it. (default: False)
  --preprocess          If set, the raw logs parameters will be preprocessed
                        and a new file created with the preprocessed logs.
                        (default: False)
  --force               force training overwriting previous output. (default:
                        False)
  --id id               Experiment id. (default: None)
  --swap                Swap testing/training data in kfold cross validation.
                        (default: False)
```



### Experiments

High level overview of each of the experiments before explaining the others

#### Testing PULearning

#### Testing Anomaly Classification on open-source datasets



### How to

Explain how to easily extend this framework and the expected outputs at each stage

#### How to run a new experiment

#### How to add a new dataset

#### How to add a new model

#### How to extract a new feature

### Citing

If you find LogClass is useful for your research, please consider citing the paper:

```
@inproceedings{meng2018device,
  title={Device-agnostic log anomaly classification with partial labels},
  author={Meng, Weibin and Liu, Ying and Zhang, Shenglin and Pei, Dan and Dong, Hui and Song, Lei and Luo, Xulong},
  booktitle={2018 IEEE/ACM 26th International Symposium on Quality of Service (IWQoS)},
  pages={1--6},
  year={2018},
  organization={IEEE}
}
```

