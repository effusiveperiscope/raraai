import os
import re
regex_pattern = r'D:\\'
regex_rep = '/mnt/data/'

def linux_filelist_line(line):
    line = line.replace('\\\\?\\', "")
    line = re.sub(regex_pattern, regex_rep, line).replace('\\','/')
    return line

def linux_filelist(filelist_path):
    with open(filelist_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
        lines = [linux_filelist_line(line) for line in lines]
        lines = list(set(lines))

    new_filelist_name = os.path.basename(filelist_path).split('.')[0] + '_linux.txt'
    new_filelist_path = os.path.join(new_filelist_name)

    with open(new_filelist_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('target')
args = parser.parse_args()

linux_filelist(args.target)
