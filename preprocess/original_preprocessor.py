from .registry import register
from .utils import process_logs, remove_parameters


def getMsgFromNewSyslog(log):

    """
    Returns log message having filtered its variables and tokens with
    non-letter characters.

    Args:
        log: Raw log message string.
    Returns:
        Filtered log message.
    """
    word_list = log.strip().split()
    msg_root = word_list[0]
    msg = " ".join(word_list[1:])
    msg = remove_parameters(msg)
    if msg:
        return msg_root + " " + msg


@register("original")
def preprocess_original(params):
    """
    Returns original logs from the paper preprocessing executor.
    """
    input_source = params['raw_logs']
    output = params['logs']
    process_logs(input_source, output, getMsgFromNewSyslog)
