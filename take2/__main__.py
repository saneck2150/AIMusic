import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Sigmoid()  # Для генерации вероятностей
        )

    def forward(self, x):
        return self.model(x)
    
class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Для определения, насколько входные данные реальны
        )

    def forward(self, x):
        return self.model(x)

#get table to Data frame format 
data = pd.read_csv("beethoven_piano_sonatas\measures\\01-1.measures.tsv", sep='\t')

#columns with string data
categorical_columns = ['timesig', 'act_dur', 'mc_offset','barline', 'breaks', 'repeats', 'next']

#extracting  columns with NaN items, and filling it with 0 or "0", depends of type of column
num_fill = {col: 0 for col, dtype in data.dtypes.items() if dtype == np.float64}
str_fill = {col: '0' for col, dtype in data.dtypes.items() if dtype == object}
fill_values = {**num_fill, **str_fill}
data.fillna(fill_values, inplace=True)

text_data = data.to_string(index=False)
encoded_data = pd.get_dummies(data)
tensor_data = torch.tensor(encoded_data.astype(np.float32).values)

# Функция потерь
adversarial_loss = nn.BCELoss()

# Инициализация генератора и дискриминатора
input_size = encoded_data.shape[1]
output_size = input_size  # Мы хотим, чтобы генератор выдавал данные того же размера, что и входные данные
generator = Generator(input_size, output_size)
discriminator = Discriminator(input_size)

# Оптимизаторы
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

num_epochs = 100
batch_size = 64
data_loader = DataLoader(TensorDataset(tensor_data), batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for real_data_batch in data_loader:
        # Данные для обучения
        real_data_batch = real_data_batch[0]
        batch_size = real_data_batch.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Обучение дискриминатора на реальных данных
        discriminator_optimizer.zero_grad()
        real_output = discriminator(real_data_batch)
        real_loss = adversarial_loss(real_output, real_labels)
        real_loss.backward()

        # Обучение дискриминатора на сгенерированных данных
        noise = torch.randn(batch_size, input_size)
        generated_data = generator(noise)
        fake_output = discriminator(generated_data.detach())
        fake_loss = adversarial_loss(fake_output, fake_labels)
        fake_loss.backward()
        discriminator_optimizer.step()

        # Обучение генератора
        generator_optimizer.zero_grad()
        fake_output = discriminator(generated_data)
        generator_loss = adversarial_loss(fake_output, real_labels)
        generator_loss.backward()
        generator_optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Generator Loss: {generator_loss.item()}, Discriminator Loss: {real_loss.item() + fake_loss.item()}')


num_samples = 1000  # количество сгенерированных примеров
noise = torch.randn(num_samples, input_size)
generated_data = generator(noise)

# Преобразование обратно в DataFrame
generated_df = pd.DataFrame(generated_data.detach().numpy(), columns=encoded_data.columns)

# Декодирование категориальных признаков
generated_df_decoded = pd.DataFrame()
for column in categorical_columns:
    # Находим индекс максимального значения в каждой строке (one-hot encoding)
    max_index = generated_df.filter(like=column).idxmax(axis=1)
    # Удаляем префикс и объединяем результаты
    decoded_column = max_index.str.replace(column + '_', '')
    generated_df_decoded[column] = decoded_column

# Восстановление пропущенных значений
generated_df_decoded.replace('0', np.nan, inplace=True)

# Вывод сгенерированных данных
print(generated_df_decoded.head())

with open('output1.txt', 'w') as file:
    file.write(generated_df_decoded.to_string())