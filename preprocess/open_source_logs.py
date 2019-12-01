from .registry import register
from .utils import remove_parameters
from tqdm import tqdm
import os


def process_line(line):
    label = line[0].strip()
    msg = ' '.join(line[1].strip().split()[1:])
    msg = remove_parameters(msg)
    if msg:
        return label + ' ' + msg


def process_open_source(input_source, output):
    with open(output, "w", encoding='latin-1') as f:
        gtruth = os.path.join(input_source, "groundtruth.seq")
        rawlog = os.path.join(input_source, "rawlog.log")
        with open(gtruth, 'r', encoding='latin-1') as IN:
            line_count = sum(1 for line in IN)
        with open(gtruth, 'r', encoding='latin-1') as in_gtruth:
            with open(rawlog, 'r', encoding='latin-1') as in_log:
                IN = zip(in_gtruth, in_log)
                for line in tqdm(IN, total=line_count):
                    result_line = process_line(line)
                    if result_line:
                        f.writelines(result_line + "\n")


open_source_datasets = [
    'open_Apache',
    'open_bgl',
    'open_hadoop',
    'open_hdfs',
    'open_hpc',
    'open_proxifier',
    'open_zookeeper',
]
for dataset in open_source_datasets:
    @register(dataset)
    def preprocess_dataset(params):
        """
        Runs open source logs preprocessing executor.
        """
        input_source = os.path.join(
            params['raw_logs'],
            dataset.split('_')[-1]
        )
        output = params['logs']
        params['healthy_label'] = 'NA'
        process_open_source(input_source, output)
