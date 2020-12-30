import sys
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms as tr
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets


MAX_EPOCH = 10
IMAGE_SIZE = 28 * 28


def main():
    train_loader = load_for_testing()

    # todo choose best model
    # model = ModelE(IMAGE_SIZE)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # for epoch in range(MAX_EPOCH):
    #     train(model, optimizer, train_loader)

    # create_test_y(model)

    # todo to comment before submit
    create_report()

    print("done")


def create_report():
    train_loader, val_loader = load_for_report()
    loss_train = {}
    loss_val = {}
    acc_train = {}
    acc_val = {}
    """
    # modelA
    print("************A**************")
    model = ModelA(IMAGE_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for e in range(10):
        print(e)
        loss_train[e], acc_train[e] = train(model, optimizer, train_loader)
        loss_val[e], acc_val[e] = validate(model, val_loader)
    
    # model B
    print("************B**************")
    model = ModelB(IMAGE_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for e in range(10):
        print(e)
        loss_train[e], acc_train[e] = train(model, optimizer, train_loader)
        loss_val[e], acc_val[e] = validate(model, val_loader)
    
    # model C
    print("************C**************")
    model = ModelC(IMAGE_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for e in range(MAX_EPOCH):
        print(e)
        loss_train[e], acc_train[e] = train(model, optimizer, train_loader)
        loss_val[e], acc_val[e] = validate(model, val_loader)
    
    # modelD
    print("************D**************")
    model = ModelD(IMAGE_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for e in range(10):
        print(e)
        loss_train[e], acc_train[e] = train(model, optimizer, train_loader)
        loss_val[e], acc_val[e] = validate(model, val_loader)
    
    # modelE
    print("************E**************")
    model = ModelE(IMAGE_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for e in range(10):
        print(e)
        loss_train[e], acc_train[e] = train(model, optimizer, train_loader)
        loss_val[e], acc_val[e] = validate(model, val_loader)
    """
    # modelF
    print("************F**************")
    model = ModelF(IMAGE_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for e in range(10):
        print(e)
        loss_train[e], acc_train[e] = train(model, optimizer, train_loader)
        loss_val[e], acc_val[e] = validate(model, val_loader)

    # Part 1 - plot average loss
    loss_graphs(loss_train, loss_val)

    # Part 2 - plot accuracy
    acc_graphs(acc_train, acc_val)

    # Part 3 - Test set accuracy
    model = ModelF(IMAGE_SIZE)
    transforms = tr.Compose([tr.ToTensor(),
                             tr.Normalize((0.1307,), (0.3081,))])
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=False, transform=transforms, download=True), batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for e in range(10):
        print(e)
        train(model, optimizer, train_loader)
        validate(model, test_loader)


def create_test_y(model):
    test_x = sys.argv[3]
    test_x = np.loadtxt(test_x)
    test_x /= 255.0
    test_x = tr.Compose([tr.ToTensor()])(test_x)[0].float()
    file = open("test_y", "w+")
    for x in test_x:
        y = model(x)
        pred = y.max(1, keepdim=True)[1].item()
        file.write(str(pred) + '\n')
    file.close()


def run_model(model, optimizer, train_loader, val_loader):
    for epoch in range(MAX_EPOCH):
        train(model, optimizer, train_loader)
        validate(model, val_loader)


def loss_graphs(avg_train_loss, avg_validation_loss):
    line1, = plt.plot(list(avg_train_loss.keys()), list(avg_train_loss.values()), "blue",
                      label='Train average Loss')
    line2, = plt.plot(list(avg_validation_loss.keys()), list(avg_validation_loss.values()), "red",
                      label='Validation average Loss')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4), line2: HandlerLine2D(numpoints=4)})
    plt.show()


def acc_graphs(avg_acc_train, avg_acc_validation):
    line1, = plt.plot(list(avg_acc_train.keys()), list(avg_acc_train.values()), "blue",
                      label='Train average Accuracy')
    line2, = plt.plot(list(avg_acc_validation.keys()), list(avg_acc_validation.values()), "red",
                      label='Validation average Accuracy')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4), line2: HandlerLine2D(numpoints=4)})
    plt.show()


def validate(model, val_loader):
    model.eval()
    correct = 0
    # loss=0
    train_loss = 0
    for data, labels in val_loader:
        labels = labels.type(torch.LongTensor)
        output = model(data)
        train_loss += F.nll_loss(output, labels)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
    train_loss /= (len(val_loader.dataset))
    acc = 100. * correct / len(val_loader.dataset)
    # todo remove print
    print('\nval set: Average loss: {:.4f}, Accuracy: {} / {}({:.0f} % )\n'.format(
            train_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))
    return train_loss, acc


def train(model, optimizer, train_loader):
    # sum_loss = 0
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
        # chek our train
        train_loss += loss
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

    train_loss /= (len(train_loader))
    acc = 100. * correct / len(train_loader.dataset)

    # todo remove print
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {} / {}({:.0f} % )\n'.format(
            train_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))

    return train_loss, acc


def load_for_report():
    train_x = np.loadtxt(sys.argv[1])
    train_y = np.loadtxt(sys.argv[2])
    # split train file to validation and train:
    size_train = int(len(train_x) * 0.2)
    validation_x = train_x[-size_train:, :]
    validation_y = train_y[-size_train:]
    train_x = train_x[: -size_train, :]
    train_y = train_y[: -size_train]

    transforms = tr.Compose([tr.ToTensor(),
                            tr.Normalize((0.1307,), (0.3081,))])
    # transforms = tr.Compose([tr.ToTensor(),
    #                          tr.Normalize((np.mean(train_x)), np.std(train_x))])
    # train data
    train_data = FashionData(train_x, train_y, transforms)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    # validation data
    validation_data = FashionData(validation_x, validation_y, transforms)
    val_loader = torch.utils.data.DataLoader(validation_data, shuffle=True)
    return train_loader, val_loader


def load_for_testing():
    train_x = np.loadtxt(sys.argv[1])
    train_y = np.loadtxt(sys.argv[2])
    transforms = tr.Compose([tr.ToTensor(),
                             tr.Normalize((0.1307,), (0.3081,))])
    # train data
    train_data = FashionData(train_x, train_y, transforms)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    return train_loader


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
        x = (self.fc2(x))
        return F.log_softmax(x, dim=1)


class ModelB(nn.Module):
    def __init__(self, image_size):
        super(ModelB, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = (self.fc2(x))
        return F.log_softmax(x, dim=1)


class ModelC(nn.Module):
    def __init__(self, image_size):
        super(ModelC, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.drop_layer = nn.Dropout(p=0.6)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = (self.fc2(x))
        self.drop_layer(x)
        return F.log_softmax(x, dim=1)


class ModelD(nn.Module):
    def __init__(self, image_size):
        super(ModelD, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.bn0 = nn.BatchNorm1d(100)
        self.fc1 = nn.Linear(100, 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 10)
        self.bn2 = nn.BatchNorm1d(10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.bn0(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        x = (self.bn2(self.fc2(x)))
        return F.log_softmax(x, dim=1)


class ModelE(nn.Module):
    def __init__(self, image_size):
        super(ModelE, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = (self.fc5(x))
        return F.log_softmax(x, dim=1)


class ModelF(nn.Module):
    def __init__(self, image_size):
        super(ModelF, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
        self.drop_layer = nn.Dropout(p=0.6)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = torch.sigmoid(self.fc0(x))
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = (self.fc5(x))
        self.drop_layer(x)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    main()
