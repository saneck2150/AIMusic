from Coder import *
from Neurolink import *
import torch.nn.functional as F

def encode_dataframe(df):
    encoded_df = df.copy()
    label_encoders = {}
    standard_scalers = {}
    
    for column in df.columns:
        if df[column].dtype == 'object':
            label_encoders[column] = LabelEncoder()
            encoded_df[column] = label_encoders[column].fit_transform(df[column])
        else:
            standard_scalers[column] = StandardScaler()
            encoded_df[column] = standard_scalers[column].fit_transform(df[[column]])
    
    tensor_data = torch.tensor(encoded_df.values, dtype=torch.float32)
    
    return tensor_data, label_encoders, standard_scalers

def decode_tensor(tensor_data, label_encoders, standard_scalers, columns):
    decoded_df = pd.DataFrame(tensor_data.numpy(), columns=columns)
    
    for column in columns:
        if column in standard_scalers:
            decoded_df[column] = standard_scalers[column].inverse_transform(decoded_df[[column]])
        elif column in label_encoders:
            decoded_df[column] = label_encoders[column].inverse_transform(decoded_df[column].astype(int))
    
    return decoded_df

def decode_dataframe(encoded_data, label_encoders, standard_scalers):
    decoded_data = pd.DataFrame()

    # Обратное преобразование для каждого столбца
    for i, column in enumerate(encoded_data.columns):
        data = encoded_data.iloc[:, i]

        # Если столбец числовой, используем обратное преобразование стандартизации
        if column in standard_scalers:
            scaler = standard_scalers[column]
            decoded_data[column] = scaler.inverse_transform(data.reshape(-1, 1)).flatten()
        # Если столбец категориальный, используем обратное преобразование кодирования меток
        elif column in label_encoders:
            encoder = label_encoders[column]
            decoded_data[column] = encoder.inverse_transform(data)

    return decoded_data


# Пример DataFrame
data = pd.read_csv("beethoven_piano_sonatas\measures\\01-1.measures.tsv", sep='\t')
df = pd.DataFrame(data)

encoded_df, label_encoders, standard_scalers = encode_dataframe(df)

input_size = encoded_df.shape[1]
output_size = input_size  # Мы хотим, чтобы генератор выдавал данные того же размера, что и входные данные
generator = Generator(input_size, output_size)
discriminator = Discriminator(input_size)

adversarial_loss = nn.BCELoss()

# Оптимизаторы
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

num_epochs = 100
batch_size = 64
data_loader = DataLoader(TensorDataset(encoded_df), batch_size=batch_size, shuffle=True)

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

        # Обучение дискриминатора на сгенерированных данных
        noise = torch.randn(batch_size, input_size)
        generated_data = generator(noise)
        fake_output = discriminator(generated_data.detach())
        discriminator_optimizer.step()

        # Обучение генератора
        generator_optimizer.zero_grad()
        fake_output = discriminator(generated_data)
        generator_optimizer.step()


noise = torch.randn(len(df), input_size)
generated_data = generator(noise)


decoded_data = decode_dataframe(generated_data.detach().numpy(), label_encoders, standard_scalers)


# В результате получаем DataFrame с данными в исходном виде
print(decoded_data)


