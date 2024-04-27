import csv
from cfg import *
from collections import Counter



# def fun (data, value_to_index):
#     index_to_value = {idx: value for value, idx in value_to_index.items()}
#     generated_table = [[index_to_value[idx] for idx in row] for row in data]
#     return generated_table


# ЗАТЕСТИТЬ !!!!
#index_to_value = {idx: value for value, idx in value_to_index.items()} \\ Table data restore
#generated_table = [[index_to_value[idx] for idx in row] for row in generated_data] \\ AI generated data restore


def getData (PATH):
    table:list[str] = []
    with open(PATH, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        for row in spamreader:
            table.append(row)
    del table[0]
    return table


#def sortTable (table:list[str], PATH:str): 
#    if PATH.find("chord") > -1:
#        transposed_table = list(map(list, zip(*table)))
#       return transposed_table

