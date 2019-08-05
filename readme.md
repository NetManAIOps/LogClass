#code for LogClass

Weibin Meng, Ying Liu, Shenglin Zhang, Dan Pei, Hui Dong, Lei Song, and Xulong Luo. Device-agnostic log anomaly classification with partial labels. Quality of Service (IWQoS), 2018 



## Please run following files one by one
### filterParameters.py
**intro**: remove parameters from raw logs. 

**run**: python filterParameter.py -rawlog ./data/rawlog.txt -output ./data/logs\_without\_paras.txt

**parameters**:

* **-rawlog**: rawlog's path
* **-output**: output path, which is the logs without paremeters.


### binary\_Classification.py
**intro**: binary classification with TF-ILF.

**run**: python binary\_Classification.py --logs ./data/logs\_without\_paras.txt  --kfold 3 --iterations 10 --prefix unlabeled --add_ilf 1

**parameters**:

* **--logs**: input path which is the output of filterParameters.py
* **--kfold**: k of kfold-crossvalidation
* **--add_ilf**: utilize TF-ILF to generate feature vector.
* **--kfold**: k of kfold-crossvalidation
* **--prefix**: the prefix of the lines of unlabeled logs. for example, the following logs are anomalous(belong to error\_2 category) and unlabeled logs respectively. You can find more logs in ./data/rawlog.txt
	*  **error\_2** [SIF pica_sif]Interface te-1/1/56, changed state to down
	*  **unlabeled** 10LLDP/5/LLDP_PVID_INCONSISTENT: -Slot=5; PVID mismatch discovered on FortyGigE5/0/12 (PVID 512), with YQ-YQ01A423-B-LY2R-135.Int qe-1/1/50 (PVID 21). 

### mululti-logClassification.py
**intro**: multiclass classification.

**run**: python mululti-logClassification.py --logs ./data/logs\_without\_paras.txt  --kfold 3 

**parameters**:

* **-logs**: input path which is the output of filterParameters.py
* **-kfold**: k of kfold-crossvalidation
* **-add_ilf**: utilize TF-ILF to generate feature vector.
