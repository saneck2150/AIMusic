import csv
from cfg import *
from collections import Counter

def tableToData (table:list[str]):
    all_values = [value for row in table for value in row]
    value_counts = Counter(all_values)
    value_to_index = {value: idx for idx, (value, _) in enumerate(value_counts.most_common())}
    numeric_table = [[value_to_index[value] for value in row] for row in table]
    return numeric_table

#index_to_value = {idx: value for value, idx in value_to_index.items()} \\ Table data restore
#generated_table = [[index_to_value[idx] for idx in row] for row in generated_data] \\ AI generated data restore

def getData (PATH):
    table:list[str] = []
    with open(PATH, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        for row in spamreader:
            table.append(row)
    del table[0]
    return tableToData(table)

#def sortTable (table:list[str], PATH:str): 
#    if PATH.find("chord") > -1:
#        transposed_table = list(map(list, zip(*table)))
#       return transposed_table


#class MusicDataset(Dataset):
#    def __init__ (self, data):  
#        self.data = data
#        
#    def __len__(self):
#        return len(self.data)
#    
#    def __getitem__(self, idx):
#        sample = self.data[idx]
#        return torch.tensor(sample)
       

