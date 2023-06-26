from typing import Callable
import sys
import re

'''
Preprocess methods I used to process JESC dataset.
The dataset contained too many weird lines so I had to removed them.
'''

def removeDecorativeCharacters(line: str):
    '''
    Delete decorative characters which are not essential for translation.
    '''
    chars_to_remove = ["\"", "", "<", ">", "(", ")", "[", "]", "。", "「", "」", "《", "》", "➡", "♪", "☎"]
    for ch in chars_to_remove:
        line = line.replace(ch, "")
    return line


def removeChineseLines(line: str):
    '''
    Delete Chinese. I think JESC dataset is not reviewed by native Japanese speaker.
    '''
    chars_likely_to_be_chinese = ["那", "你", "请", "节", "违", "这"]
    if line.startswith("我"):
        return ""
    for ch in chars_likely_to_be_chinese:
        if ch in line:
            return ""
    return line


def removeSpeakerName(line: str):
    '''
    Delete speaker's name in Japanese sentence.

    ex. "(オリヴィエ)" in this sentence:
    (オリヴィエ)うん。 おっ。 パンデピス?
    '''
    line = re.sub("[\(\[].*?[\)\]]", "", line)
    return line


def Preprocess(filename: str, output: str, *prcs: Callable[[str], str]):
    in_file = open(filename, "r")
    out_file = open(output, "w")
    while True:
        line = in_file.readline()
        if not line:
            break
        for process in prcs:
            line = process(line)
        split_line = line.split("\t")
        if len(split_line) != 2:
            continue
        # Japanese part contains \n
        split_line[1] = split_line[1].replace("\n", "")

        if len(split_line[0]) != 0 and len(split_line[1]) != 0:
            out_file.write(line)


if __name__ == "__main__":
    if sys.argv.__len__() < 2:
        print(sys.argv[0] + " dataset_root_dir_name")
        exit()
    if not sys.argv[1].endswith("/"):
        sys.argv[1] += "/"
    dirname = sys.argv[1]
    Preprocess(dirname + "train", dirname + "train_p", removeSpeakerName,
               removeDecorativeCharacters, removeChineseLines)
    Preprocess(dirname + "dev", dirname + "dev_p", removeSpeakerName, removeDecorativeCharacters, removeChineseLines)
