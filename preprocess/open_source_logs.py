import os
from multiprocessing import Pool

from tqdm import tqdm

from .registry import register
from .utils import remove_parameters


def process_line(line):
    label = line[0].strip()
    msg = " ".join(line[1].strip().split()[1:])
    msg = remove_parameters(msg)
    if msg:
        msg = " ".join((label, msg))
        msg = "".join((msg, "\n"))
        return msg
    return ""


def process_open_source(input_source, output):
    with open(output, "w", encoding="latin-1") as f:
        gtruth = os.path.join(input_source, "groundtruth.seq")
        rawlog = os.path.join(input_source, "rawlog.log")
        with open(gtruth, "r", encoding="latin-1") as IN:
            line_count = sum(1 for line in IN)
        with open(gtruth, "r", encoding="latin-1") as in_gtruth:
            with open(rawlog, "r", encoding="latin-1") as in_log:
                IN = zip(in_gtruth, in_log)
                with Pool() as pool:
                    results = pool.imap(process_line, IN, chunksize=10000)
                    f.writelines(tqdm(results, total=line_count))


open_source_datasets = [
    "open_Apache",
    "open_bgl",
    "open_hadoop",
    "open_hdfs",
    "open_hpc",
    "open_proxifier",
    "open_zookeeper",
]
for dataset in open_source_datasets:

    @register(dataset)
    def preprocess_dataset(params):
        """
        Runs open source logs preprocessing executor.
        """
        input_source = params["raw_logs"]
        output = params["logs"]
        params["healthy_label"] = "NA"
        process_open_source(input_source, output)
