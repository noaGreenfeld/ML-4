import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms as tr
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets


# todo change name of function and remove redundant print
def print_hi(name):
    print(f'Hi, {name}')
    train_loader, val_loader = load()
    #modela
    model = ModelA(28*28)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for e in range(10):
        print(e)
        train(model, optimizer, train_loader)
        validate(model, val_loader)

    print("done")
def validate(model, val_loader):
    model.eval()
    correct=0
    loss=0
    train_loss = 0
    for data, labels in (val_loader):
        labels = labels.type(torch.LongTensor)
        output = model(data)
        loss += F.nll_loss(output, labels)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
    train_loss /= (len(val_loader.dataset))
    print('\nval set: Average loss: {:.4f}, Accuracy: {} / {}({:.0f} % )\n'.format(
    train_loss, correct, len(val_loader.dataset),
    100. * correct / len(val_loader.dataset)))

def train(model, optimizer, train_loader):
        sum_loss = 0
        correct = 0
        train_loss = 0
        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            #
            labels = labels.type(torch.LongTensor)
            #
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
            #chek our train
            sum_loss+=loss
            pred =output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
        train_loss /= (len(train_loader.dataset))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {} / {}({:.0f} % )\n'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))


def load():
    train_x = np.loadtxt("train_x")
    train_y = np.loadtxt("train_y")
    # split train file to validation and train:
    size_train = int(len(train_x) * 0.2)
    validation_x = train_x[-size_train:, :]
    validation_y = train_y[-size_train:]
    train_x = train_x[: -size_train, :]
    train_y = train_y[: -size_train]

    # todo - check the avg and change it
    transforms = tr.Compose([tr.ToTensor(),
                            tr.Normalize((0.1307,), (0.3081,))])
    # train data
    train_data = FashionData(train_x, train_y, transforms)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    # validation data
    validation_data = FashionData(validation_x, validation_y, transforms)
    val_loader = torch.utils.data.DataLoader(validation_data, shuffle=True)
    return train_loader, val_loader


class FashionData(Dataset):
    def __init__(self, x, y, transforms):
        self.x = x
        self.y = y
        self.transforms = transforms

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        data = self.x[idx, :]
        data = np.asarray(data).astype(np.uint8).reshape(28,28)
        if self.transforms:
             data = self.transforms(data)
        if self.y is not None:
            return data, self.y[idx]
        else:
            return data

class ModelA(nn.Module):
    def __init__(self, image_size):
        super(ModelA, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

class ModelB(nn.Module):
    def __init__(self, image_size):
        super(ModelA, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)
if __name__ == '__main__':
    print_hi('noaaaaa')

