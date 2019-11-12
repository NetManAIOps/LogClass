from .registry import register
from .utils import process_logs, remove_parameters
import re


recid_regx = re.compile(r"^(\d+)")
separator = re.compile("(?:-.{1,3}){2} (.+)$")
msg_split_regx = re.compile(r"x'.+'")
severity = re.compile(r"(INFO|WARN|ERROR|FATAL)")


def process_line_bgl(line):
    line = line.strip()
    sep = separator.search(line)
    if sep:
        msg = sep.group(1).strip().split('   ')[-1].strip()
        msg = msg_split_regx.split(msg)[-1].strip()
        error_label = severity.search(line)
        recid = recid_regx.search(line)
        if recid and error_label and len(msg) > 20:
            # recid = recid.group(1).strip() We may want to use it later
            label = error_label.group(1)
            if label == 'WARN':
                return None
            if label == 'INFO':  # or label == 'WARN':
                label = 'unlabeled'
            msg = remove_parameters(msg)
            if msg:
                return label + ' ' + msg


@register("bgl")
def preprocess_bgl(params):
    """
    Runs BGL logs preprocessing executor.
    """
    input_source = params['raw_logs']
    output = params['logs']
    params['healthy_label'] = 'unlabeled'
    process_logs(input_source, output, process_line_bgl)
