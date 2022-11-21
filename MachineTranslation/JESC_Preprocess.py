from typing import Callable

def removeDoubleQuote(line:str):
    return line.replace("\"","")

def PreProcess(filename:str, output:str, *prcs:Callable[[str],str]):
    in_file=open(filename,"r")
    out_file=open(output,"w")
    while True:
        line=in_file.readline()
        if not line:
            break
        for process in prcs:
            line=process(line)
        out_file.write(line)

if __name__ == "__main__":
    PreProcess("dataset/train","dataset/train_p",removeDoubleQuote)
    PreProcess("dataset/dev","dataset/dev_p",removeDoubleQuote)
