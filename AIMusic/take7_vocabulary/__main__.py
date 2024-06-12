import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

data = pd.read_csv("beethoven_piano_sonatas/measures/32-2.measures.tsv", sep='\t')
df = pd.DataFrame(data)
df = df.fillna(0)
list_of_lists = df.values.tolist()
transposed_list = list(map(list, zip(*list_of_lists)))
unique_values_dict = {df.columns[i]: set(transposed_list[i]) for i in range(len(df.columns))}

numbered_list = []
for sublist, col_name in zip(transposed_list, df.columns):
    numbered_sublist = [list(unique_values_dict[col_name]).index(item) for item in sublist]
    numbered_list.append(numbered_sublist)

print(numbered_list)
numbered_list = list(map(list, zip(*numbered_list)))
##########################################################

data = torch.tensor(numbered_list, dtype=torch.float32)
# Определение простой RNN модели
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        out = self.relu(out)
        return out

# Параметры модели
input_size = len(numbered_list[0])  # Количество признаков в данных
hidden_size = 150  # Размер скрытого состояния LSTM
output_size = input_size  # Размер выхода модели

# Создание модели
model = LSTMModel(input_size, hidden_size, output_size)

# Определение функции потерь и оптимизатора
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Обучение модели
num_epochs = 25000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(data.unsqueeze(0))  # Добавляем размерность batch_size=1
    loss = criterion(output[:, :-1], data.unsqueeze(0)[:, 1:])  # Сравниваем все предсказанные значения, кроме первого, с истинными данными
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Генерация новых значений с помощью обученной модели
model.eval()
with torch.no_grad():
    output = model(data.unsqueeze(0))  # Предсказание для всей последовательности данных
    generated_seq = output.squeeze(0).numpy()  # Преобразуем тензор в numpy array
    generated_seq = np.maximum(generated_seq, 0)  # Принудительно ограничиваем значения неотрицательными

print("Сгенерированная последовательность:")
rounded_generated_seq = [[int(round(value)) for value in row] for row in generated_seq]

print("Сгенерированная последовательность с округленными значениями:")
print(rounded_generated_seq)

##########################################################
new_data_list = list(map(list, zip(*rounded_generated_seq)))
original_list_of_lists = []
for col_idx, sublist in enumerate(new_data_list):
         column_name = list(unique_values_dict.keys())[col_idx]
         column_values = list(unique_values_dict[column_name])
         original_sublist = [
            column_values[item] if 0 <= item < len(column_values) else 0 
            for item in sublist]
         original_list_of_lists.append(original_sublist)

        
original_list_of_lists = list(map(list, zip(*original_list_of_lists)))
print ("end")
for raw in original_list_of_lists:
    print(raw)
