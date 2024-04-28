from Dataset import *
from Neurolink import *
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import re

# input parameters +
PATH = 'beethoven_piano_sonatas\measures\\01-1.measures.tsv'
table_measures = getData(PATH) #list[list[str]]
numeric_data = []
text_data = []


# 1. Подготовка данных
for row in table_measures:
    numeric_row = []
    text_row = []
    # Проход по каждому элементу в строке
    for item in row:
        # Проверка, является ли элемент числом
        if re.match(r'^-?\d+\.?\d*$', item):
            numeric_row.append(float(item))  # Преобразование строки в число
        else:
            text_row.append(item)  # Добавление элемента в текстовые данные
    numeric_data.append(numeric_row)
    text_data.append(text_row)

print("Numeric data:")
print(numeric_data)
print("Text data:")
print(text_data)

# Преобразование текстовых данных в числовые признаки с помощью Bag-of-Words
vectorizer = CountVectorizer()
X_text_numeric = vectorizer.fit_transform(text_data).toarray()
X_combined = np.concatenate((numeric_data, X_text_numeric), axis=1)










# class CustomDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.float32)
    
#     def __len__(self):
#         return len(self.X)
    
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

# # Загрузка данных
# # Пример: X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
# train_dataset = CustomDataset(X_train, y_train)
# test_dataset = CustomDataset(X_test, y_test)

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64)

# 2. Определение модели нейронной сети
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# # 3. Обучение модели
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * inputs.size(0)
#     epoch_loss = running_loss / len(train_loader.dataset)
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# # 4. Оценка модели
# model.eval()
# test_loss = 0.0
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         test_loss += loss.item() * inputs.size(0)
# test_loss = test_loss / len(test_loader.dataset)
# print(f'Test Loss: {test_loss:.4f}')

# 5. Генерация таблиц
# После того, как модель обучена, можешь использовать ее для генерации предсказаний на новых данных
# И затем использовать эти предсказания для создания новых таблиц











# # input parameters +
# PATH = 'beethoven_piano_sonatas\measures\\01-1.measures.tsv'
# table_measures = getData(PATH) #list[list[str]]

# for row in table_measures:
#     print(row)

# ########## table vectorisation +            ### Проблема может быть тут, гдето не так векторизирую данные
# table_text = [' '.join(row) for row in table_measures] #to list[str]
# vectorizer = TfidfTransformer()
# vectorised_table = vectorizer.fit_transform(table_text)
# dense_matrix = vectorised_table.toarray()
# print(vectorised_table)

# ########## vectorised table to tenzor transformation + 
# inputs = torch.tensor(dense_matrix, dtype=torch.float)

# ######### neurolink parameters +
# input_size_measures = inputs.shape[1]  #Number of columns in input table
# hidden_size_measures = 20 #Number of neurons
# output_size_measures = input_size_measures #Number of columns in output table
# num_epochs = 10

# ########## model connection +
# model = Autoencoder(input_size_measures, hidden_size_measures)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# ########## neurolink study +
# for epoch in range(num_epochs):
#     optimizer.zero_grad()
#     outputs = model(inputs)
#     loss = criterion(outputs, inputs)  # Сравниваем выход с входом
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# latent_space_samples = torch.randn(10, hidden_size_measures)  # Генерация случайных скрытых представлений
# generated_data = model.decoder(latent_space_samples)

# # Преобразование тензоров PyTorch обратно в numpy массивы
# generated_data_numpy = generated_data.detach().numpy()

# # Вывод сгенерированных данных
# print("Generated data:")
# print(generated_data_numpy)

# # Обратное преобразование из матрицы векторов в изначальный формат таблицы
# generated_table_text = vectorizer.inverse_transform(generated_data_numpy)

# dense_matrix_back = inputs.numpy()

# # Преобразование плотной матрицы в список списков строк
# table_measures_back = []
# for row in dense_matrix_back:
#     formatted_row = [str(cell) for cell in row]
#     table_measures_back.append(formatted_row)
