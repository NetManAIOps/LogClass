import re
import numpy as np

def remove_parameters(msg):
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
    return msg


def process_logs(input_source, output, process_line=None):
    with open(output, "w") as f:
        with open(input_source, 'r', encoding='latin-1') as IN:
            for line in IN:
                result_line = process_line(line)
                if result_line:
                    f.writelines(result_line + "\n")


def load_logs(log_path, unlabel_label='unlabeled', ignore_unlabeled=False):
    x_data = []
    y_data = []
    label_dict = {}
    target_names = []
    with open(log_path) as IN:
        for line in IN:
            L = line.strip().split()
            label = L[0]
            if label not in label_dict:
                if ignore_unlabeled and label == unlabel_label:
                    continue
                if label == unlabel_label:
                    label_dict[label] = -1.0
                elif label not in label_dict:
                    label_dict[label] = len(label_dict)
                    target_names.append(label)
            x_data.append(" ".join(L[1:]))
            y_data.append(label_dict[label])
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    return x_data, y_data, target_names
