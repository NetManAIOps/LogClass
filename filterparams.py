#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
Reads rawlog file and outputs filtered logs removing parameter words
or tokens with non-letter characters keeping only text words.
"""


import re
import argparse


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        help="input file path",
        type=str,
        default="./LogClass/data/rawlog.txt"
    )
    parser.add_argument(
        "--output",
        help="output file path",
        type=str,
        default="./LogClass/data/logs_without_paras.txt",
    )
    args = parser.parse_args()
    input_filename = args.input
    output_filename = args.output
    with open(output_filename, "w") as f:
        with open(input_filename) as IN:
            for line in IN:
                nen = getMsgFromNewSyslog(line)
                if len(nen.split()) <= 1:
                    continue
                f.writelines(nen + "\n")
    print("rawlogs:" + input_filename)
    print("Parameters have been removed")
    print("logs without parameters:" + output_filename)
