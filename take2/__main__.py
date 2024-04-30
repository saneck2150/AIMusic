from Neurolink import *

#get table to Data frame format 
df = pd.read_csv("beethoven_piano_sonatas/measures/04-1.measures.tsv", sep='\t')
df_str = df.astype(str)
# Инициализация кодировщика OneHotEncoder
encoder = OneHotEncoder()
categorical_columns = ['mc', 'mn', 'quarterbeats', 'duration_qb', 'keysig', 'timesig', 'act_dur', 'mc_offset', 'numbering_offset', 'dont_count', 'barline', 'breaks', 'repeats', 'next']
# Преобразование текстовых данных в OHE и преобразование в numpy array
encoded_df = encoder.fit_transform(df[categorical_columns]).toarray()
# Преобразование в тензор PyTorch
tensor_df = torch.tensor(encoded_df, dtype=torch.float32)
print(tensor_df)

# Функция потерь
adversarial_loss = nn.BCELoss()

# Инициализация генератора и дискриминатора
input_size = tensor_df.shape[1]
output_size = input_size  # Мы хотим, чтобы генератор выдавал данные того же размера, что и входные данные
generator = Generator(input_size, output_size)
discriminator = Discriminator(input_size)

# Оптимизаторы
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.002)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

num_epochs = 200
batch_size = 64
data_loader = DataLoader(TensorDataset(tensor_df), batch_size=batch_size, shuffle=True)

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

decoded_data = generated_data.detach().numpy()
# # # Обратное преобразование OHE в исходные категории
decoded_categories = encoder.inverse_transform(decoded_data)
# Создание нового DataFrame из декодированных категорий
decoded_df = pd.DataFrame(decoded_categories, columns=categorical_columns)
print(decoded_df)

with open('output1.txt', 'w') as file:
    file.write(decoded_df.to_string())