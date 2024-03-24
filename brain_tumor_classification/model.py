import torch
from torch import nn
from torch.nn import functional as F
from hyperparams import INPUT_SIZE, CLASSES


class TumorClassificationCNN_FC(nn.Module):  # 2
    def __init__(self):
        super().__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # output dim: in + 2 * padding - filter + 1
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        # linear layers
        self.fc1 = nn.Linear(16 * (INPUT_SIZE // 4) * (INPUT_SIZE // 4), 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 4)
        # dropout
        self.dropout = nn.Dropout(p=0.2)
        # max pooling
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  #  8 x 64 x 64
        x = self.pool(F.relu(self.conv2(x)))  # 16 x 32 x 32
        # TODO több réteg, legalább 5 réteg, nem kell annxi FC a végére
        # flattening the image
        x = torch.flatten(x, 1)
        # linear layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.fc5(x)
        # TODO log softmax
        return x


class TumorClassificationCNN_Conv(nn.Module):  # 1
    def __init__(self) -> None:
        super().__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # output dim: in + 2 * padding - filter + 1
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(16, 8, 3, padding=1)
        # linear layers
        self.fc = nn.Linear(8 * 4 * 4, len(CLASSES))
        # max pooling
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  #  8 x 64 x 64
        x = self.pool(F.relu(self.conv2(x)))  # 16 x 32 x 32
        x = self.pool(F.relu(self.conv3(x)))  # 32 x 16 x 16
        x = self.pool(F.relu(self.conv4(x)))  # 16 x 8  x 8
        x = self.pool(F.relu(self.conv5(x)))  #  8 x 4  x 4

        # flattening the image
        # x = torch.flatten(x, 1)
        x = x.view(-1, 128)
        # linear layer
        x = F.log_softmax(self.fc(x), dim=1)
        return x


class TumorClassificationCNN_Mixed(nn.Module):  # 3
    def __init__(self):
        super().__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)  # output dim: in + 2 * padding - filter + 1
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
        # linear layers
        self.fc1 = nn.Linear(16 * (INPUT_SIZE // 8) * (INPUT_SIZE // 8), 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 4)
        # dropout
        self.dropout = nn.Dropout(p=0.2)
        # max pooling
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  #  4 x 64 x 64
        x = self.pool(F.relu(self.conv2(x)))  #  8 x 32 x 32
        x = self.pool(F.relu(self.conv3(x)))  # 16 x 16 x 16
        # flattening the image
        x = torch.flatten(x, 1)
        # linear layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x


class TumorClassificationCNN_Conv_Norm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # output dim: in + 2 * padding - filter + 1
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(16, 8, 3, padding=1)
        # linear layers
        self.fc = nn.Linear(8 * 4 * 4, len(CLASSES))
        # max pooling
        self.pool = nn.MaxPool2d(2, 2)
        # batch norm
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self, x):
        # convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  #  8 x 64 x 64
        x = self.pool(F.relu(self.bn1(self.conv2(x))))  # 16 x 32 x 32
        x = self.pool(F.relu(self.conv3(x)))  # 32 x 16 x 16
        x = self.pool(F.relu(self.bn2(self.conv4(x))))  # 16 x 8  x 8
        x = self.pool(F.relu(self.conv5(x)))  #  8 x 4  x 4

        # flattening the image
        # x = torch.flatten(x, 1)
        x = x.view(-1, 128)
        # linear layer
        x = F.log_softmax(self.fc(x), dim=1)
        return x


class TumorClassificationCNN_Mixed_Norm(nn.Module):
    def __init__(self):
        super().__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)  # output dim: in + 2 * padding - filter + 1
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(8)
        # linear layers
        self.fc1 = nn.Linear(16 * (INPUT_SIZE // 8) * (INPUT_SIZE // 8), 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 4)
        # dropout
        self.dropout = nn.Dropout(p=0.1)
        # max pooling
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  #  4 x 64 x 64
        x = self.pool(F.relu(self.bn1(self.conv2(x))))  #  8 x 32 x 32
        x = self.pool(F.relu(self.conv3(x)))  # 16 x 16 x 16
        # flattening the image
        x = torch.flatten(x, 1)
        # linear layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x


class DogClassificationCNN(nn.Module):
    def __init__(self):
        super(DogClassificationCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.bn1 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 30 * 30, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, len(CLASSES))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.bn1(self.conv2(x))))
        x = x.view(-1, 32 * 30 * 30)
        x = F.relu(self.bn2(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
