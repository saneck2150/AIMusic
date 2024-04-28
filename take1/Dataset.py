import csv
from cfg import *
from collections import Counter


def getData (PATH):
    table:list[str] = []
    with open(PATH, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        for row in spamreader:
            table.append(row)
    del table[0]
    return table


