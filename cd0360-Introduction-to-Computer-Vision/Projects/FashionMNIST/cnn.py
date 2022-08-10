import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Each image is a 28x28 grayscale image
        # First convolution layer: 1 input image channel (grayscale), 10 output channels/feature maps
        # 3 X 3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 10, 3)

        # max pooling layer: kernel size = 2, stride size = 2
        self.pool = nn.MaxPool2d(2, 2)

        # Second convolution layer
        self.conv2 = nn.Conv2d(10, 20, 3)

        # Fully connected layer
        self.fc1 = nn.Linear(20 * 5 * 5, 50)

        # Dropout with p = 0.4
        self.fc1_drop = nn.Dropout(p=0.4)

        # Finally, created the 10 output channels (for the 10 classes)
        self.fc2 = nn.Linear(50, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    net = Net()
    print(net)
