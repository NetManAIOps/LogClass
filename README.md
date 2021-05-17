## LogClass
This repository provides an open-source toolkit for LogClass framework from W. Meng et al., "[LogClass: Anomalous Log Identification and Classification with Partial Labels](https://ieeexplore.ieee.org/document/9339940)," in IEEE Transactions on Network and Service Management, doi: 10.1109/TNSM.2021.3055425.

LogClass automatically and accurately detects and classifies anomalous logs based on partial labels.

### Table of Contents

[LogClass](#logclass)

- [Table of Contents](#table-of-contents)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
	- [Run LogClass](#run-logclass)
	- [Arguments](#arguments)
	- [Directory Structure](#directory-structure)
	- [Datasets](#datasets)
- [How to](#how-to)
	- [How to add a new dataset](#how-to-add-a-new-dataset)
		- [Preprocessed Logs Format](#preprocessed-logs-format)
	- [How to run a new experiment](#how-to-run-a-new-experiment)
		- [Custom experiment](#custom-experiment)
	- [How to add a new model](#how-to-add-a-new-model)
	- [How to extract a new feature](#how-to-extract-a-new-feature)
- [Included Experiments](#included-experiments)
	- [Testing PULearning](#testing-pulearning)
	- [Testing Anomaly Classification](#testing-anomaly-classification)
	- [Global LogClass](#global-logclass)
	- [Binary training/inference](#binary-traininginference)
- [Citing](#citing)

​		

### Requirements

Requirements are listed in `requirements.txt`. To install these, run:

```
pip install -r requirements.txt
```



### Quick Start

#### Run LogClass

Several example experiments using LogClass are included in this repository. 

Here is an example to run one of them -  training of the global experiment doing anomaly detection and classification.  Run the following command in the home directory of this project: 

```
python -m LogClass.logclass --train --kfold 3 --logs_type "bgl" --raw_logs "./Data/RAS_LOGS" --report macro
```



#### Arguments

```
python -m LogClass.logclass --help
usage: logclass.py [-h] [--raw_logs raw_logs] [--base_dir base_dir]
                   [--logs logs] [--models_dir models_dir]
                   [--features_dir features_dir] [--logs_type logs_type]
                   [--kfold kfold] [--healthy_label healthy_label]
                   [--features features [features ...]]
                   [--report report [report ...]]
                   [--binary_classifier binary_classifier]
                   [--multi_classifier multi_classifier] [--train] [--force]
                   [--id id] [--swap]

Runs binary classification with PULearning to detect anomalous logs.

optional arguments:
  -h, --help            show this help message and exit
  --raw_logs raw_logs   input raw logs file path (default: None)
  --base_dir base_dir   base output directory for pipeline output files
                        (default: ['{your_logclass_dir}\\output'])
  --logs logs           input logs file path and output for raw logs
                        preprocessing (default: None)
  --models_dir models_dir
                        trained models input/output directory path (default:
                        None)
  --features_dir features_dir
                        trained features_dir input/output directory path
                        (default: None)
  --logs_type logs_type
                        Input type of logs. (default: ['open_Apache'])
  --kfold kfold         kfold crossvalidation (default: None)
  --healthy_label healthy_label
                        the labels of unlabeled logs (default: ['unlabeled'])
  --features features [features ...]
                        Features to be extracted from the logs messages.
                        (default: ['tfilf'])
  --report report [report ...]
                        Reports to be generated from the model and its
                        predictions. (default: None)
  --binary_classifier binary_classifier
                        Binary classifier to be used as anomaly detector.
                        (default: ['pu_learning'])
  --multi_classifier multi_classifier
                        Multi-clas classifier to classify anomalies. (default:
                        ['svm'])
  --train               If set, logclass will train on the given data.
                        Otherwiseit will run inference on it. (default: False)
  --force               Force training overwriting previous output with same
                        id. (default: False)
  --id id               Experiment id. Automatically generated if not
                        specified. (default: None)
  --swap                Swap testing/training data in kfold cross validation.
                        (default: False)
```



#### Directory Structure

```
.
├── data
│   └── open_source_logs		# Included open-source log datasets
│       ├── Apache
│       ├── bgl
│       ├── hadoop
│       ├── hdfs
│       ├── hpc
│       ├── proxifier
│       └── zookeeper
├── output				# Example output folder
│   ├── preprocessed_logs		# Saved preprocessed logs for reuse
│   │   ├── open_Apache.txt
│   │   └── open_bgl.txt
│   └── train_multi_open_bgl_2283696426	# Example experiment output
│       ├── best_params.json
│       ├── features
│       │   ├── tfidf.pkl
│       │   └── vocab.pkl
│       ├── models
│       │   └── multi.pkl
│       └── results.csv
├── feature_engineering
│   ├── __init__.py
│   ├── length.py
│   ├── tf_idf.py
│   ├── tf_ilf.py
│   ├── tf.py
│   ├── registry.py
│   ├── vectorizer.py			# Log message vectorizing utilities
│   └── utils.py
├── models
│   ├── __init__.py
│   ├── base_model.py			# BaseModel class extended by all models
│   ├── pu_learning.py
│   ├── regular.py
│   ├── svm.py
│   ├── binary_registry.py
│   └── multi_registry.py
├── preprocess
│   ├── __init__.py
│   ├── bgl_preprocessor.py
│   ├── open_source_logs.py
│   ├── registry.py
│   └── utils.py
├── reporting
│   ├── __init__.py
│   ├── accuracy.py
│   ├── confusion_matrix.py
│   ├── macrof1.py
│   ├── microf1.py
│   ├── multi_class_acc.py
│   ├── top_k_svm.py
│   ├── bb_registry.py
│   └── wb_registry.py
├── puLearning				# PULearning third party implementation
│   ├── __init__.py
│   └── puAdapter.py
├── __init__.py
├── LICENSE
├── README.md
├── requirements.txt
├── init_params.py			# Parses arguments, initializes global parameters
├── logclass.py				# Performs training and inference of LogClass
├── test_pu.py				# Compares robustness of LogClass
├── train_multi.py			# Trains LogClass for anomalies classification
├── train_binary.py			# Trains LogClass for log anomaly detection
├── run_binary.py			# Loads trained LogClass and detects anomalies
├── decorators.py
└── utils.py
```

#### Datasets

In this repository we include various [open-source logs datasets](https://github.com/logpai/loghub) in the `data` folder as well as their corresponding preprocessing module in the `preprocess` package. Additionally there is  another preprocessor provided for [BGL logs data](https://www.usenix.org/cfdr-data#hpc4), which can be downloaded directly from [here](https://www.usenix.org/sites/default/files/4372-intrepid_ras_0901_0908_scrubbed.zip.tar).



### How to

Explain how to use and extend this toolkit.

#### How to add a new dataset

Add a new preprocessor module in the `preprocess` package.

The module should implement a function that follows the `preprocess_datset(params)` function template included in all preprocessors. It should be decorated with `@register(f"{dataset_name}")` , e.g. open_Apache, and call the `process_logs(input_source, output, process_line)` function. This `process_line` function should also be defined in the processor as well. 

When done, add the module name to the `__init__.py`  list of modules from the `preprocess` package and also the name from the decorator in the argsparse parameters options as the logs type. For example,  `--logs_type open_Apache`.

##### Preprocessed Logs Format

This format is ensured by the `process_line` function which is to be defined in each preprocessor.

```python
def process_line(line):
    """ 
    Processes a given line from the raw logs.

    Parameter
    ---------
    line : str
        One line from the raw logs.

    Returns
    -------
    str
        String with the format f"{label} {msg}" where the `label` indicates whether
        the log is anomalous and if so, which anomaly category, and `msg` is the
        filtered log message without parameters.

    """
# your code
```

To filter the log message parameters, use the `remove_parameters(msg)`function from the `utils.py` module in the `preprocess` package.

#### How to run a new experiment

Several experiments examples are included in the repository.  The best way to start with creating a new one is to follow the example from the others, specially the main function structure and its experiment function be it training or testing focused.

The key things to consider the experiment should include are the following:

- **Args parsing**:  create custom `init_args()` and `parse_args(args)` functions for your experiment that call `init_main_args()` from the `init_params.py` module.

- **Output file handling**: use `file_handling(params)` function (see `utils.py` in the main directory of the repo).

- **Preprocessing raw logs**: if `--raw_logs` argument is provided, get the preprocessing function using the `--logs_type` argument from the `preprocess` module registry calling `get_preprocessor(f'{logs_type}')` function.

- **Load logs**: call the `load_logs(params, ...)` function to get the preprocessed logs from the directory specified in the `--logs` parameter. It will return a tuple  of x, y, and target label names data.


##### Custom experiment

Main functions to consider for a custom experiment. Usually in its own function.

**Feature Engineering**

- `extract_features(x, params)` from `feature_engineering` package's `utils.py` module: Extracts all specified features in `--features` parameter from the preprocessed logs. See the function definition for further details.
- `build_vocabulary(x)` from `feature_engineering` package's `vectorizer.py` module: Divides log into tokens and creates vocabulary. See the function definition for further details.
- `log_to_vector(x, vocabulary)` from `feature_engineering` package's `vectorizer.py` module: Vectorizes each log message using a dict of words to index. See the function definition for further details.
- `get_features_vector(x_vector, vocabulary, params)` from `feature_engineering` package's `utils.py` module: Extracts all specified features from the vectorized logs. See the function definition for further details.



**Model training and inference**

Each model extends the `BaseModel` class from module `base_model.py`.  See the class definition for further details.

There are two registries in the `models` package, one for binary models meant to be used for anomaly detection and another one for multi-classification models to classify the anomalies. Get the constructor for either using the `--binary_classifier` or `--multi_classifier` argument specified. E.g. `binary_classifier_registry.get_binary_model(params['binary_classifier'])`.

By extending `BaseModel` the model is always saved when it fits the data. Load a model by calling its `load()` method. It will use the `params` attribute of the `BaseModel` class to get the experiment id and load its corresponding model. 

To save the params of an experiment call the `save_params(params)` function from the `utils.py` module in the main directory. `load_params(params)` in case of only using the module for inference. 

**Reporting**

There are two kinds of reports, black box and white box and a registry for each in the `reporting` module.

To use them, call the corresponding registry and obtain the report wrapper using `black_box_report_registry.get_bb_report('acc')`, for example. 

To add new reports, see the analogous explanation for [models](#how-to-add-a-new-model) or [features](#how-to-extract-a-new-feature) below.

**Saving results**

Among the provided experiments, `test_pu.py` and `train_multi.py` save their results creating a dict of column names to lists of results. Then the `save_results.py` function from the `utils.py` module is used to save them to a CSV file.



#### How to add a new model

To add a new model, implement a class that extends the `BaseModel` class and include its module in the `models` package. See the class definition for further details. 

Decorate a method that calls its constructor and returns an instance of the model with the `@register(f"{model_name}")`decorator from either the `binary_registry.py` or the `multi_registry.py` modules from the `models` package depending on whether the model is for anomaly detection or classification respectively. 

Finally, make sure you add the module's name in the `__init__.py` module from the `models` package and the model option in the `init_params.py` module within the list for either `--binary_classifier` or `multi_classifier` arguments. This way the constructor can be obtained by doing `binary_classifier_registry.get_binary_model(params['binary_classifier'])`, for example.



#### How to extract a new feature

To add a new feature extractor, create a module in the `feature_engineering` package that wraps your feature extractor function and returns the features. See `length.py` module as an example for further details.

As in the other cases, decorate the wrapper function with `@register(f"{feature_name}")` and make sure you add the module name in the `__init__.py` from the `feature_engineering` package and the feature as an option in the `init_params.py` module `--features` argument.



### Included Experiments

High level overview of each of the experiments included in the repository.  

#### Testing PULearning

`test_pu.py` is mainly focused on proving the robustness of LogClass for anomaly detection when just providing few labeled data as anomalous.

It would compare PULearning+RandomForest with any other given anomaly detection algorithm. Using the given data, it would start with having only healthy logs on the unlabeled data and gradually increase this up to 10%. To test PULearning, run the following command in the home directory of this project: 

```
python -m LogClass.test_pu --logs_type "bgl" --raw_logs "./Data/RAS from Weibin/RAS_raw_label.dat" --binary_classifier regular --ratio 8 --step 1 --top_percentage 11 --kfold 3
```

This would first preprocess the logs. Then, for each kfold iteration, it will perform feature extraction and force a 1:8 ratio of anomalous:healthy logs. Finally with a step of 1% it will go from 0% to 10% anomalous logs in the unlabeled set and compare the accuracy of both anomaly detection algorithms. If none specified it will default to a plain RF. 

#### Testing Anomaly Classification

`train_multi.py` is focused on showing the robustness of LogClass' TF-ILF feature extraction approach for multi-class anomaly classification. The main detail is that when using `--kfold N`, one can swap training/testing data slices using the `--swap` flag. This way, for instance, it can train on 10% logs and test on the remaining 90%, when pairing `--swap` with n ==10. To run such an experiment, use the following command from the parent directory of the project:

```
python -m LogClass.train_multi --logs_type "open_Apache" --raw_logs "./Data/open_source_logs/" --kfold 10 --swap
```

#### Global LogClass

`logclass.py` is set up so that it does both training or testing of the learned models depending on the flags. For example to train and preprocessing run the following command in the home directory of this project: :

```
python -m LogClass.logclass --train --kfold 3 --logs_type "bgl" --raw_logs "./Data/RAS_LOGS" 
```

This would first preprocess the raw BGL logs and extract their TF-ILF features, then train and save both PULearning with a RandomForest for anomaly detection and an SVM for multi-class anomaly classification. 

For running inference simply run:

```
python -m LogClass.logclass --logs_type 
```

In this case it would load the learned feature extraction approach, both learned models and run inference on the whole logs.

#### Binary training/inference

`train_binary.py` and `run_binary.py` simply separate the binary part of `logclass.py` into two modules: one for training both feature extraction and the models, and another one for loading these and running inference.



### Citing

If you find LogClass is useful for your research, please consider citing the paper:

```
@ARTICLE{9339940,  author={Meng, Weibin and Liu, Ying and Zhang, Shenglin and Zaiter, Federico and Zhang, Yuzhe and Huang, Yuheng and Yu, Zhaoyang and Zhang, Yuzhi and Song, Lei and Zhang, Ming and Pei, Dan},
journal={IEEE Transactions on Network and Service Management},
title={LogClass: Anomalous Log Identification and Classification with Partial Labels},
year={2021},
doi={10.1109/TNSM.2021.3055425}
}
```

This code was completed by [@Weibin Meng](https://github.com/WeibinMeng) and [@Federico Zaiter](https://github.com/federicozaiter).
