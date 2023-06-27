import sys
from typing import Callable

"""
Preprocess methods to process JParaCrawl dataset(https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/)

The dataset is a TSV file with following format:
en_domain   jp_domain   bicleaner_score en_sentence jp_sentence
(I don't really know if it's correct or not cause there's no official info about the format)

As a side note, I'll leave some basic information about the dataset.

JParaCrawl is a subproject of ParaCrawl, a project whose objective is to create parallel corpora from the web.
Original ParaCrawl is mainly focused on Indo-European languages, and Asian languages are not actively supported.

ParaCrawl consists of three steps:
1. Crawl the Web
2. Align documents
3. Align sentences

First, ParaCrawl finds multi-lingual domain by looking at the language detected in the said domain, then crawl the candidate domains.
Then, the documents(web pages) are aligned in pairs using some heuristic methods(bilingual lexica, HTML structure, Image contents, URL).
After documents are aligned, texts on those documents are splitted into sentences, then they are aligned in pairs again.

JParaCrawl claims to work in the exact same way as ParaCrawl does, but the paper doesn't explain how they align documents.

"""


def JParaCrawl_Preprocess(filename: str, output: str, *prcs: Callable[[list[str]], list[str]]):
    in_file = open(filename, "r")
    out_file = open(output, "w")
    count = 0
    while True:
        count += 1
        line = in_file.readline()
        split_line = line.split("\t")
        if not line:
            break
        for process in prcs:
            split_line = process(split_line)
            if not split_line:
                break

        if len(split_line) != 5:
            continue

        concated_line = "{}\t{}".format(split_line[3], split_line[4])
        out_file.write(concated_line)


def FilterLowScores(split_line: list[str]):
    if float(split_line[2]) < 0.65:
        split_line = []
    return split_line


def EscapeDoubleQuote(split_line: list[str]):
    split_line[3] = split_line[3].replace('"', '\\"')
    split_line[4] = split_line[4].replace('"', '\\"')
    return split_line


def RemoveTabs(split_line: list[str]):
    split_line[3] = split_line[3].replace("\t", " ")
    split_line[4] = split_line[4].replace("\t", " ")
    return split_line


if __name__ == "__main__":
    if sys.argv.__len__() == 1:
        print(sys.argv[0] + " input_filepath output_filepath")
        exit()
    input_filepath = sys.argv[1]
    output_filepath = "processed.tsv"
    if sys.argv.__len__() > 2:
        output_filepath = sys.argv[2]

    JParaCrawl_Preprocess(input_filepath, output_filepath, FilterLowScores, RemoveTabs, EscapeDoubleQuote)
