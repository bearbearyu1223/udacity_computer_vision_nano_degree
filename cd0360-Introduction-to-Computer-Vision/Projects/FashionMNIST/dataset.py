from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

batch_size = 20


def load_data(batch_size=batch_size, classes=None, verbose=True, visualize=False):
    if classes is None:
        classes = classes
    data_transform = transforms.ToTensor()
    train_data = FashionMNIST(root="./data", train=True, download=True, transform=data_transform)
    test_data = FashionMNIST(root="./data", train=False, download=True, transform=data_transform)
    if verbose:
        print("Train Data: number of images: {}".format(len(train_data)))
        print("Test Data: number of images: {}".format(len(test_data)))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    if visualize:
        dataiter = iter(train_loader)
        images, labels = dataiter.next()
        images = images.numpy()
        fig = plt.figure(figsize=(25, 4))
        for idx in np.arange(batch_size):
            ax = fig.add_subplot(2, batch_size / 2, idx + 1, xticks=[], yticks=[])
            ax.imshow(np.squeeze(images[idx]), cmap="gray")
            ax.set_title(classes[labels[idx]])
        plt.show()
    return train_loader, test_loader


if __name__ == "__main__":
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    load_data(classes=classes, visualize=True)
