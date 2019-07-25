#code for LogClass

Weibin Meng, Ying Liu, Shenglin Zhang, Dan Pei, Hui Dong, Lei Song, and Xulong Luo. Device-agnostic log anomaly classification with partial labels. Quality of Service (IWQoS), 2018 



## Please run following files one by one
### filterParameters.py
**intro**: remove parameters from raw logs. 

**run**: python filterParameter.py -rawlog ./data/rawlog.txt -output ./data/logs\_without\_paras.txt

**parameters**:

* -rawlog: rawlog's path
* -output: output path, which is the logs without paremeters.


### ilf\_binary\_Classification.py
**intro**: binary classification with TF-ILF

**run**: python ilf\_binary\_Classification.py -logs ./data/logs\_without\_paras.txt  -kfold 3 -iterations 10 -unlabel unlabeled -add_ilf 1

**parameters**:

* -logs: input path, logs without paremeters
* -output: output path, which is the logs without paremeters.

