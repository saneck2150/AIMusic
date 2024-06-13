from __import__ import *
from __neurolink__ import *
from __functional__ import *

# DATA PREPARATION
input_data = pd.read_csv("beethoven_piano_sonatas/measures/32-2.measures.tsv", sep='\t')
df = pd.DataFrame(input_data)
df = df.fillna(0)
list_of_lists = df.values.tolist()
numbered_list, unique_values_dict = encodeList(list_of_lists, df)

#MODEL DEFINITION
data = torch.tensor(numbered_list, dtype=torch.float32)

input_size = len(numbered_list[0]) 
hidden_size = 150
output_size = input_size

model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

#MODEL STUDYNG
num_epochs = 25000
x_axis = []
y_axis = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(data.unsqueeze(0))  # Добавляем размерность batch_size=1
    loss = criterion(output[:, :-1], data.unsqueeze(0)[:, 1:])  # Сравниваем все предсказанные значения, кроме первого, с истинными данными
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        x_axis.append(epoch)
        y_axis.append((loss.item()))
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

plt.plot(x_axis, y_axis)
plt.show()

#NEW DATA GENERATOR
model.eval()
with torch.no_grad():
    output = model(data.unsqueeze(0))  # Предсказание для всей последовательности данных
    generated_seq = output.squeeze(0).numpy()  # Преобразуем тензор в numpy array
    generated_seq = np.maximum(generated_seq, 0)  # Принудительно ограничиваем значения неотрицательными
rounded_generated_seq = [[int(round(value)) for value in row] for row in generated_seq]

#OUTPUT DATA PRPARATION       
original_list_of_lists = decodeList(rounded_generated_seq, unique_values_dict)
# print ("end")
# for raw in original_list_of_lists:
#     print(raw)

trans_original_list_of_lists = list(map(list, zip(*original_list_of_lists)))
trans_numbered_list = list(map(list, zip(*numbered_list)))
trans_numbered_list[0].pop(0)
trans_numbered_list[0].append(0)
real_seq = data.numpy()
nll_real = list(map(list, zip(*real_seq)))

nll_loss, kl_div, diversity = calculate_metrics(trans_original_list_of_lists, trans_numbered_list)

print(f'NLL Loss: {nll_loss:.4f}')
print(f'KL Divergence: {kl_div:.4f}')
print(f'Diversity: {diversity:.4f}')

#OUTPUT
with open("output1.tsv", 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerows(original_list_of_lists)