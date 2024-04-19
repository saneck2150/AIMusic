import torch
from Dataset import *

table = getData(PATH)
transpose_table = sortTable(table, PATH)
print(transpose_table[0])

