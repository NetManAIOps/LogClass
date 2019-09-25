import re


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
    # Removing parameters with Regex
    msg = re.sub("(:(?=\s))|((?<=\s):)", "", msg)
    msg = re.sub("(\d+\.)+\d+", "", msg)
    msg = re.sub("\d{2}:\d{2}:\d{2}", "", msg)
    msg = re.sub("Mar|Apr|Dec|Jan|Feb|Nov|Oct|May|Jun|Jul|Aug|Sep", "", msg)
    msg = re.sub(":?(\w+:)+", "", msg)
    msg = re.sub("\.|\(|\)|\<|\>|\/|\-|\=|\[|\]", " ", msg)
    L = msg.split()
    p = re.compile("[^(A-Za-z)]")
    # Filtering strings that have non-letter tokens
    new_msg = [k for k in L if not p.search(k)]
    msg = " ".join(new_msg)
    return msg_root + " " + msg
