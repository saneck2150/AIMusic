import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

PATH = 'beethoven_piano_sonatas\chords\\01-1.chords.tsv'

table:list[str] = []
with open(PATH, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
    for row in spamreader:
        table.append(row)

        

