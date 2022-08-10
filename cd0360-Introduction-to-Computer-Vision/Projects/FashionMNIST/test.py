from dataset import load_data, batch_size
from cnn import Net
import matplotlib
import numpy as np
import os
import torch
import torch.nn as nn

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_loader, test_loader = load_data(classes=classes)
model_name = 'fashion_net_ex.pt'
model_dir = 'saved_models'
model_file_path = os.path.join(os.getcwd(), model_dir, model_name)
net = Net()
net.load_state_dict(torch.load(model_file_path))
criterion = nn.CrossEntropyLoss()


def test_eval(test_loader, model, criterion, visualize=False):
    model.eval()
    test_loss = torch.zeros(1)
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    for batch_i, data in enumerate(test_loader):
        inputs, labels = data
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss = test_loss + ((torch.ones(1) / (batch_i + 1)) * (loss.data - test_loss))
        _, predicted = torch.max(outputs.data, 1)
        correct = np.squeeze(predicted.eq(labels.data.view_as(predicted)))

        for i in range(batch_size):
            label = labels.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    print('Test Loss: {:.6f}\n'.format(test_loss.numpy()[0]))

    for i in range(len(classes)):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

    if visualize:
        dataiter = iter(test_loader)
        images, labels = dataiter.next()
        preds = np.squeeze(model(images).data.max(1, keepdim=True)[1].numpy())
        images = images.numpy()
        fig = plt.figure(figsize=(25, 4))
        for idx in np.arange(batch_size):
            ax = fig.add_subplot(2, batch_size / 2, idx + 1, xticks=[], yticks=[])
            ax.imshow(np.squeeze(images[idx]), cmap='gray')
            ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                         color=("green" if preds[idx] == labels[idx] else "red"))
        plt.show()


if __name__ == "__main__":
    test_eval(test_loader=test_loader, model=net, criterion=criterion, visualize=True)

