import torch
import numpy as np
import matplotlib.pyplot as plt

# Генерация большего объема данных
train_size = 1000
test_size = 100

train_data = torch.Tensor(np.random.randint(10, size=train_size))
test_data = torch.Tensor(np.random.randint(10, size=test_size))

train_data = train_data.view(train_size, 1)
test_data = test_data.view(test_size, 1)

train_target = torch.Tensor(np.sin(train_data.numpy()))
test_target = torch.Tensor(np.sin(test_data.numpy()))
#Создаем слои
class NeuralNet(torch.nn.Module):
    def __init__(self, n1, n2, n3, n4):
        super(NeuralNet, self).__init__()
        self.l1 = torch.nn.Linear(1, n1)
        self.act1 = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(n1, n2)
        self.act2 = torch.nn.ReLU()
        self.l3 = torch.nn.Linear(n2, n3)
        self.act3 = torch.nn.ReLU()
        self.l4 = torch.nn.Linear(n3, n4)

    def forward(self, x):
        x = self.act1(self.l1(x))
        x = self.act2(self.l2(x))
        x = self.act3(self.l3(x))
        x = self.l4(x) 
        return x

net = NeuralNet(20, 32, 20, 1)
#Оптимизируем через метод Adam на основе потерь 
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
# для функция потерь цел и потерянные 
def loss(y, t):
    s = (y - t) ** 2
    return s.mean()

# Обучение модели на большем объеме данных
for i in range(1000):
    optimizer.zero_grad()
    out = net(train_data)
    e = loss(out, train_target)
    #минимализирует ошибки
    e.backward()
    optimizer.step()

# Прогнозирование на тестовых данных
predicted_test = net(test_data)

# Вычисление потерь на тестовых данных
test_loss = loss(predicted_test, test_target)

# Графф
predicted_test = predicted_test.cpu().detach().numpy()

plt.figure(figsize=(10, 5))
plt.plot(predicted_test, color='red', label='Ответы нейронной сети')
plt.plot(test_target, color='green', label='Правильные ответы')
plt.legend()

plt.text(1.01, 0.5, f'Test Loss: {test_loss.item():.4f}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)

plt.show()