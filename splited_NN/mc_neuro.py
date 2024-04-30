from __include__ import *

#Neuro for mc
class mcNet(nn.Module):
    def __init__(self):
        super(mcNet, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        x = self.fc(x)
        return x

net = mcNet()

def mc_training ():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    input_data = torch.tensor(data[:-1], dtype=torch.float).view(-1, 1)
    target_data = torch.tensor(data[1:], dtype=torch.float).view(-1, 1)
    # Обучение нейронной сети
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    epochs = 10000
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = net(input_data)
        loss = criterion(outputs, target_data)
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(f'mc Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

def mc_next_generate (current):
    # Генерация нового ряда на основе обученной нейронной сети
    with torch.no_grad():
    # Прогнозируем следующее число на основе последнего в данных
        new_input = torch.tensor(current, dtype=torch.float).view(-1, 1)
        predicted_value = net(new_input).item()
    return predicted_value