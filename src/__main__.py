from Dataset import *
from Neurolink import *
from sklearn.feature_extraction.text import CountVectorizer

# input parameters +
PATH = 'beethoven_piano_sonatas\measures\\01-1.measures.tsv'
table_measures = getData(PATH) #list[list[str]]

for row in table_measures:
    print(row)

########## table vectorisation + 
table_text = [' '.join(row) for row in table_measures] #to list[str]
vectorizer = CountVectorizer()
vectorised_table = vectorizer.fit_transform(table_text)
dense_matrix = vectorised_table.toarray()
print(vectorised_table)

########## vectorised table to tenzor transformation + 
inputs = torch.tensor(dense_matrix, dtype=torch.float)

######### neurolink parameters +
input_size_measures = inputs.shape[1]  #Number of columns in input table
hidden_size_measures = 20 #Number of neurons
output_size_measures = input_size_measures #Number of columns in output table
num_epochs = 10

########## model connection +
model = Autoencoder(input_size_measures, hidden_size_measures)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

########## neurolink study +
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, inputs)  # Сравниваем выход с входом
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

latent_space_samples = torch.randn(10, hidden_size_measures)  # Генерация случайных скрытых представлений
generated_data = model.decoder(latent_space_samples)

# Преобразование тензоров PyTorch обратно в numpy массивы
generated_data_numpy = generated_data.detach().numpy()

# Вывод сгенерированных данных
print("Generated data:")
print(generated_data_numpy)

# Обратное преобразование из матрицы векторов в изначальный формат таблицы
generated_table_text = vectorizer.inverse_transform(generated_data_numpy)

dense_matrix_back = inputs.numpy()

# Преобразование плотной матрицы в список списков строк
table_measures_back = []
for row in dense_matrix_back:
    formatted_row = [str(cell) for cell in row]
    table_measures_back.append(formatted_row)
