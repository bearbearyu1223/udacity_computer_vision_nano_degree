from cnn import Net
from dataset import load_data
import torch
import torch.optim as optim
import torch.nn as nn
import datetime
import matplotlib
import os

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def train(n_epochs, train_loader, visualize=False):
    loss_over_time = []
    print("Start Training: {}".format(datetime.datetime.now()))
    for epoch in range(n_epochs):
        running_loss = 0.0
        for batch_i, data in enumerate(train_loader):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss = running_loss + loss.item()

            if batch_i % 1000 == 999:
                avg_loss = running_loss / 1000
                loss_over_time.append(avg_loss)
                print("Epoch: {}, Batch: {}, Avg_Loss: {}".format(epoch + 1, batch_i + 1, avg_loss))
                running_loss = 0.0
    print("Finish Training: {}".format(datetime.datetime.now()))
    if visualize:
        plt.plot(loss_over_time)
        plt.xlabel('1000\'s of batches')
        plt.ylabel('loss')
        plt.ylim(0, 2.5)  # consistent scale
        plt.show()
    return loss_over_time


if __name__ == "__main__":
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    train_loader, test_loader = load_data(classes=classes)
    criterion = nn.CrossEntropyLoss()
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    n_epochs = 30
    loss_over_time = train(n_epochs=n_epochs, train_loader=train_loader)
    model_dir = os.path.join(os.getcwd(), 'saved_models')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_name = 'fashion_net_ex.pt'
    torch.save(net.state_dict(), "saved_models/" + model_name)


