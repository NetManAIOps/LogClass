#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : Weibin Meng
# * Email         : m_weibin@163.com
# * Create time   : 2017-09-13 12:32
# * Last modified : 2019-05-28 11:45
# * Filename      : filterParameter.py
# * Description   :
'''
'''
# **********************************************************

import os
import re


def getMsgFromNewSyslog(log):

    '''
    '''
    word_list = log.strip().split()
    msg_id=word_list[0]
    msg_root=word_list[0]
    msg=' '.join(word_list[1:])
    msg = re.sub('(:(?=\s))|((?<=\s):)', '', msg)
    msg = re.sub('(\d+\.)+\d+', '', msg)
    msg = re.sub('\d{2}:\d{2}:\d{2}', '', msg)
    msg = re.sub('Mar|Apr|Dec|Jan|Feb|Nov|Oct|May|Jun|Jul|Aug|Sep', '', msg)
    msg = re.sub(':?(\w+:)+', '', msg)
    msg = re.sub('\.|\(|\)|\<|\>|\/|\-|\=|\[|\]',' ',msg)
    l = msg.split()
    p = re.compile('[^(A-Za-z)]')
    new_msg = []
    for k in l:
        m = p.search(k)
        if m:
            continue
        else:
            new_msg.append(k)
    msg = ' '.join(new_msg)
    return msg_root+' '+msg

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_filename', help = 'input_file', type = str, default = '../data/rawlog_1000.txt')
    parser.add_argument('-output_filename', help = 'output_file', type = str, default = '../data/logs_without_paras_1000.txt')
    args = parser.parse_args()
    input_filename = args.input_filename
    output_filename = args.output_filename
    f = open(output_filename,'w')
    with open(input_filename) as IN:
        for line in IN:
            nen = getMsgFromNewSyslog(line)
            if len(nen.split())<=1:
                continue
            f.writelines(nen+'\n')
    print('finished~~~~~')
