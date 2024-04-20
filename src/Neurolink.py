from Dataset import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def dataToTensor (table:list[str]):
    max_length = max(len(row) for row in table)
    padded_table = [row + [''] * (max_length - len(row)) for row in table]
    tensor_data = []
    for row in padded_table:
        numeric_row = []
        for item in row:
            try:
                numeric_row.append(float(item))
            except ValueError:
                numeric_row.append(0.0)
        tensor_data.append(numeric_row)

    for row in tensor_data:
        for item in row:
            print(item)

    tensor_data = torch.tensor(tensor_data)

    print("Размерность тензора:", tensor_data.size())
    return tensor_data