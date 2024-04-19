import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from cfg import *

def getData (PATH):
    table:list[str] = []
    with open(PATH, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        for row in spamreader:
            table.append(row)
    return table

def sortTable (table:list[str], PATH:str):
    if PATH.find("chord") > -1:
        transposed_table = list(map(list, zip(*table)))
        return transposed_table


class MusicDataset(Dataset):
    def __init__ (self, data):  
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.tensor(sample)
       

