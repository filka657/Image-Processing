import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import numpy as np


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(32*32, 512)  # 32x32 512
        self.fc2 = nn.Linear(512, 256)  # 512 256
        self.fc3 = nn.Linear(256, 128)  # 256 128
        self.fc4 = nn.Linear(128, 64)  # 128 64
        self.fc5 = nn.Linear(64, 32)  # 64 32
        self.fc6 = nn.Linear(32, 10)  # 32 16

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)  # Relu method
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        x = nn.ReLU()(x)
        x = self.fc4(x)
        x = nn.ReLU()(x)
        x = self.fc5(x)
        x = nn.ReLU()(x)
        x = self.fc6(x)
        return x


def calculate_metrics(loader, model):
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.numpy())
            y_true.extend(labels.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return accuracy, recall, precision, f1


transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
valset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

net = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=0.0003)

accuracy = []
recall = []
precision = []
fone_score = []


def creating_plots(accuracy, recall, precision, fone_score):
    fig, axes = plt.subplots(2, 2)
    fig.suptitle('Метрики валидации')
    axes[0, 0].plot([int(i) for i in range(10)], accuracy)
    axes[0, 0].set_title('Accuracy')
    axes[0, 1].plot([int(i) for i in range(10)], recall)
    axes[0, 1].set_title('Recall')
    axes[1, 0].plot([int(i) for i in range(10)], precision)
    axes[1, 0].set_title('Precision')
    axes[1, 1].plot([int(i) for i in range(10)], fone_score)
    axes[1, 1].set_title('F1 score')

    plt.show()


for epoch in range(10):
    running_loss = 0.0
    net.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.3f}')

    train_acc, train_rec, train_prec, train_f1 = calculate_metrics(trainloader, net)
    print(f'Training - Accuracy: {train_acc}, Recall: {train_rec}, Precision: {train_prec}, F1 Score: {train_f1}')

    val_acc, val_rec, val_prec, val_f1 = calculate_metrics(valloader, net)
    accuracy.append(val_acc)
    recall.append(val_rec)
    precision.append(val_prec)
    fone_score.append(val_f1)
    print(f'Validation - Accuracy: {val_acc}, Recall: {val_rec}, Precision: {val_prec}, F1 Score: {val_f1}')

creating_plots(accuracy, recall, precision, fone_score)
print('Finished Training')
