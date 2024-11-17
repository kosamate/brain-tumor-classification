import abc
import torch
from torch import nn
from torch.nn import functional as F


class TumorClassification(nn.Module, abc.ABC):
    def __init__(self, input_size: int, class_count: int):
        super().__init__()
        self._input_size = input_size
        self._class_count = class_count


class TC_CNN_FC(TumorClassification):  # 2
    def __init__(self, input_size: int, class_count: int):
        super().__init__(input_size, class_count)
        # convolutional layers
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # output dim: in + 2 * padding - filter + 1
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        # linear layers
        self.fc1 = nn.Linear(16 * (self._input_size // 4) * (self._input_size // 4), 1024)
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
        # flattening the image
        x = torch.flatten(x, 1)
        # linear layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.fc5(x)
        return x


class TC_CNN_Conv(TumorClassification):  # 1
    def __init__(self, input_size: int, class_count: int) -> None:
        super().__init__(input_size, class_count)
        # convolutional layers
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # output dim: in + 2 * padding - filter + 1
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(16, 8, 3, padding=1)
        # linear layers
        self.fc = nn.Linear(8 * 4 * 4, self._class_count)
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


class TC_CNN_Mixed(TumorClassification):  # 3
    def __init__(self, input_size: int, class_count: int):
        super().__init__(input_size, class_count)
        # convolutional layers
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)  # output dim: in + 2 * padding - filter + 1
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
        # linear layers
        self.fc1 = nn.Linear(16 * (self._input_size // 8) * (self._input_size // 8), 1024)
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


class TC_CNN_Conv_Norm(TumorClassification):
    def __init__(self, input_size: int, class_count: int) -> None:
        super().__init__(input_size, class_count)
        # convolutional layers
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # output dim: in + 2 * padding - filter + 1
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(16, 8, 3, padding=1)
        # linear layers
        self.fc = nn.Linear(8 * 4 * 4, self._class_count)
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


class TC_CNN_Mixed_Norm(TumorClassification):
    def __init__(self, input_size: int, class_count: int):
        super().__init__(input_size, class_count)
        # convolutional layers
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)  # output dim: in + 2 * padding - filter + 1
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(8)
        # linear layers
        self.fc1 = nn.Linear(16 * (self._input_size // 8) * (self._input_size // 8), 1024)
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


class DogClassificationCNN(TumorClassification):
    def __init__(self, input_size: int, class_count: int):
        super().__init__(input_size, class_count)

        self.conv1 = nn.Conv2d(1, 16, 3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.bn1 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 30 * 30, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self._class_count)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.bn1(self.conv2(x))))
        x = x.view(-1, 32 * 30 * 30)
        x = F.relu(self.bn2(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TC_CNN_Normal(TumorClassification):
    def __init__(self, input_size: int, class_count: int):
        super().__init__(input_size, class_count)

        self.conv1 = nn.Conv2d(1, 16, 3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.bn1 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 30 * 30, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self._class_count)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.bn1(self.conv2(x))))
        x = x.view(-1, 32 * 30 * 30)
        x = F.relu(self.bn2(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


class TC_CNN_Big(TumorClassification):
    def __init__(self, input_size: int, class_count: int):
        super().__init__(input_size, class_count)

        self.conv1 = nn.Conv2d(1, 16, 3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.bn0 = nn.BatchNorm2d(32)
        self.bn1 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self._class_count)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.bn0(self.conv2(x))))
        x = self.pool(F.relu(self.bn1(self.conv3(x))))
        x = x.view(-1, 64 * 14 * 14)
        x = F.relu(self.bn2(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


class TC_Final(TumorClassification):
    def __init__(self, input_size: int, class_count: int):
        super().__init__(input_size, class_count)

        self.conv32 = nn.Conv2d(1, 32, 3, stride=1)
        self.conv32to64 = nn.Conv2d(32, 64, 3, stride=1)

        self.conv64to64 = nn.Conv2d(64, 64, 3, stride=1)
        self.conv64to128 = nn.Conv2d(64, 128, 3, stride=1)

        self.conv128to128 = nn.Conv2d(128, 128, 3, stride=1)
        self.conv256 = nn.Conv2d(128, 256, 3, stride=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.bn64 = nn.BatchNorm2d(64)
        self.bn128 = nn.BatchNorm2d(128)
        self.bn256 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(6400, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, self._class_count)

    def forward(self, x):
        # input 148 x 148
        x = F.relu(self.conv32(x))
        x = F.relu(self.bn64(self.conv32to64(x)))
        x = self.pool(x)  # output 72x72

        x = F.relu(self.conv64to64(x))
        x = F.relu(self.bn64(self.conv64to64(x)))
        x = self.pool(x)  # output 34x34

        x = F.relu(self.conv64to128(x))
        x = F.relu(self.conv128to128(x))
        x = F.relu(self.bn128(self.conv128to128(x)))
        x = self.pool(x)  # output 14x14

        x = F.relu(self.conv128to128(x))
        x = F.relu(self.bn256(self.conv256(x)))
        x = self.pool(x)  # output: 5x5

        x = x.view(-1, 6400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)

        return x


class TC_Final_D(TumorClassification):
    def __init__(self, input_size: int, class_count: int):
        super().__init__(input_size, class_count)

        self.conv32 = nn.Conv2d(1, 32, 3, stride=1)
        self.conv32to64 = nn.Conv2d(32, 64, 3, stride=1)

        self.conv64to64 = nn.Conv2d(64, 64, 3, stride=1)
        self.conv64to128 = nn.Conv2d(64, 128, 3, stride=1)

        self.conv128to128 = nn.Conv2d(128, 128, 3, stride=1)
        self.conv256 = nn.Conv2d(128, 256, 3, stride=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.35)

        self.fc1 = nn.Linear(6400, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, self._class_count)

    def forward(self, x):
        # input 148 x 148
        x = F.relu(self.conv32(x))
        x = F.relu(self.conv32to64(x))
        x = self.pool(x)  # output 72x72
        x = self.dropout(x)

        x = F.relu(self.conv64to64(x))
        x = F.relu(self.conv64to64(x))
        x = self.pool(x)  # output 34x34
        x = self.dropout(x)

        x = F.relu(self.conv64to128(x))
        x = F.relu(self.conv128to128(x))
        x = F.relu(self.conv128to128(x))
        x = self.pool(x)  # output 14x14
        x = self.dropout(x)

        x = F.relu(self.conv128to128(x))
        x = F.relu(self.conv256(x))
        x = self.pool(x)  # output: 5x5
        x = self.dropout(x)

        x = x.view(-1, 6400)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)

        return x
