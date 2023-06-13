from typing import Callable
import sys


def removeDoubleQuote(line: str):
    return line.replace("\"", "")


def PreProcess(filename: str, output: str, *prcs: Callable[[str], str]):
    in_file = open(filename, "r")
    out_file = open(output, "w")
    while True:
        line = in_file.readline()
        if not line:
            break
        for process in prcs:
            line = process(line)
        out_file.write(line)


if __name__ == "__main__":
    if sys.argv.__len__() < 2:
        print(sys.argv[0] + " dataset_root_dir_name")
        exit()
    if not sys.argv[1].endswith("/"):
        sys.argv[1] += "/"
    dirname = sys.argv[1]
    PreProcess(dirname + "train", dirname + "train_p", removeDoubleQuote)
    PreProcess(dirname + "dev", dirname + "dev_p", removeDoubleQuote)
