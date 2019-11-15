from .registry import register
from .utils import process_logs, remove_parameters


def process_line_bgl(line):
    line = line.strip().split()
    general_label = line[0]
    if general_label == 'INFO':
        label = 'unlabeled'
    else:
        label = line[1]
    msg = ' '.join(line[2:])
    msg = remove_parameters(msg)
    if msg:
        return label + ' ' + msg


@register("bgl_old")
def preprocess_bgl(params):
    """
    Runs BGL logs preprocessing executor.
    """
    input_source = params['raw_logs']
    output = params['logs']
    params['healthy_label'] = 'unlabeled'
    process_logs(input_source, output, process_line_bgl)
