## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your NaimishNet
import torch.nn.init as I


class NaimishNet(nn.Module):

    def __init__(self):
        super(NaimishNet, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        # first input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # output size = (W - F)/S + 1 = (224 - 5)/ 1 + 1 = 220
        # after one maxpool layer, the output tensor will have the dimensions: (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(p=0.1)
        self.batch_norm1 = nn.BatchNorm2d(32)

        # second conv layer: 32 inputs, 64 outputs, 5X5 conv
        # output size = (W -F)/S + 1 = (110 - 5) + 1 = 106
        # after another pool layer, the output tensor will have dimensions : (64, 54, 54)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(p=0.1)
        self.batch_norm2 = nn.BatchNorm2d(64)

        # third conv layer: 64 inputs, 128 outputs, 5X5 conv
        # output size = (W -F)/S + 1 = (54 - 5) + 1 = 50
        # after another pool layer, the output tensor will have dimensions : (128, 25, 25)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(p=0.1)
        self.batch_norm3 = nn.BatchNorm2d(128)

        # fourth conv layer: 128 inputs, 256 outputs, 5X5 conv
        # output size = (W -F)/S + 1 = (25 - 5) + 1 = 21
        # after another pool layer, the output tensor will have dimensions : (256, 10, 10)
        self.conv4 = nn.Conv2d(128, 256, 5)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout(p=0.2)
        self.batch_norm4 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256 * 10 * 10, 1024)
        self.relu5 = nn.ReLU()
        self.drop5 = nn.Dropout(p=0.2)
        self.batch_norm5 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 1024)
        self.relu6 = nn.ReLU()
        self.drop6 = nn.Dropout(p=0.2)
        self.batch_norm6 = nn.BatchNorm1d(1024)

        # finally, created the 68 output channels (for the 68 keypoints)
        self.fc3 = nn.Linear(1024, 68*2)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.batch_norm1(self.drop1(self.pool1(self.relu1(self.conv1(x)))))
        x = self.batch_norm2(self.drop2(self.pool2(self.relu2(self.conv2(x)))))
        x = self.batch_norm3(self.drop3(self.pool3(self.relu3(self.conv3(x)))))
        x = self.batch_norm4(self.drop4(self.pool4(self.relu4(self.conv4(x)))))
        x = x.view(x.size(0), -1)

        x = self.batch_norm5(self.drop5(self.relu5(self.fc1(x))))
        x = self.batch_norm6(self.drop6(self.relu6(self.fc2(x))))
        x = self.fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
