from __include__ import *
# Neuro for quarterbeats and duration_qb prediction
# На основе quarterbeats генерируется duration_qb
class qbNet(nn.Module):
    def __init__(self):
        super(qbNet, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = qbNet()
    
def duration_qb_training ():
    quarterbeats_main = np.array([0,1,3,5,6,8,12,13,15])
    quarterbeats_semi = np.array([1,3,5,6,8,12,13,15,15])
    duration_qb = np.array([1,2,2,1,2,4,1,2,0])

    # Преобразование в тензоры PyTorch
    X = torch.tensor(np.column_stack((quarterbeats_main, quarterbeats_semi)), dtype=torch.float32)
    y = torch.tensor(duration_qb, dtype=torch.float32).unsqueeze(1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Обучение модели
    num_epochs = 1000
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
    
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def duration_qb_generate (val):
    with torch.no_grad():
        test_input = torch.tensor([val], dtype=torch.float32)
        predicted_duration = model(test_input)
    return float(predicted_duration)