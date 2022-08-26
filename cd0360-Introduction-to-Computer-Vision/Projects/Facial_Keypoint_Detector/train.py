import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataset import FacialKeypointsDataset, Rescale, RandomCrop, Normalize, ToTensor
from models import NaimishNet
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


def create_transformed_train_test_dataset(train_csv_file_path, train_image_dir,
                                          test_csv_file_path, test_image_dir, transform, verbose=False):
    train_dataset = FacialKeypointsDataset(csv_file_path=train_csv_file_path, image_dir=train_image_dir,
                                           transform=transform)
    test_dataset = FacialKeypointsDataset(csv_file_path=test_csv_file_path, image_dir=test_image_dir,
                                          transform=transform)
    return train_dataset, test_dataset


def create_train_test_dataloader(train_dataset, test_dataset, batch_size, shuffle=True, num_workers=0):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return train_loader, test_loader


def test_net_sample_output(test_loader, model):
    for i, sample in enumerate(test_loader):
        images = sample["image"]
        key_pts = sample["keypoints"]

        images = images.type(torch.FloatTensor)
        output_pts = model(images)

        output_pts = output_pts.view(output_pts.size()[0], 68, -1)

        if i == 0:
            return images, output_pts, key_pts


def show_all_key_points(image, predicted_key_pts, ground_truth_pts=None):
    plt.imshow(image, cmap="gray")
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker=".", c="m")
    if ground_truth_pts is not None:
        plt.scatter(ground_truth_pts[:, 0], ground_truth_pts[:, 1], s=20, marker=".", c="g")


def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):
    plt.figure(figsize=(20, 10))
    for i in range(batch_size):
        ax = plt.subplot(1, batch_size, i + 1)

        image = test_images[i].data
        image = image.numpy()
        image = np.transpose(image, (1, 2, 0))

        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.numpy()
        predicted_key_pts = predicted_key_pts * 50 + 100

        # plot ground truth points for comparison, if they exist
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]
            ground_truth_pts = ground_truth_pts * 50 + 100

        # call show_all_keypoints
        show_all_key_points(np.squeeze(image), predicted_key_pts, ground_truth_pts)
        plt.axis('off')
    plt.show()


def train(model, n_epochs, train_loader):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    model.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        for batch_i, data in enumerate(train_loader):
            images = data["image"]
            key_pts = data["keypoints"]
            key_pts = key_pts.view(key_pts.size(0), -1)
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            output_pts = model(images)

            loss = criterion(output_pts, key_pts)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if batch_i % 10 == 9:  # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i + 1, running_loss/10))
                running_loss = 0.0
    print('Finished Training')


if __name__ == "__main__":
    root_dir = os.path.join(os.getcwd(), "data")
    train_csv_file_name = "training_frames_keypoints.csv"
    test_csv_file_name = "test_frames_keypoints.csv"
    train_csv_file_path = os.path.join(root_dir, train_csv_file_name)
    test_csv_file_path = os.path.join(root_dir, test_csv_file_name)
    train_image_dir = os.path.join(root_dir, "training/")
    test_image_dir = os.path.join(root_dir, "test/")
    transform = transforms.Compose([Rescale((224, 224)), Normalize(), ToTensor()])
    train_dataset, test_dataset = create_transformed_train_test_dataset(train_csv_file_path, train_image_dir,
                                                                        test_csv_file_path, test_image_dir, transform,
                                                                        verbose=False)
    batch_size = 16
    train_loader, test_loader = create_train_test_dataloader(train_dataset, test_dataset,
                                                             batch_size=batch_size, shuffle=True, num_workers=0)
    model = NaimishNet()

    n_epochs = 5  # start small, and increase when you've decided on your model structure and hyperparams
    train(model, n_epochs, train_loader)
    test_images, test_outputs, gt_pts = test_net_sample_output(test_loader, model)
    visualize_output(test_images, test_outputs, gt_pts)

    model_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'keypoints_model_1.pt'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    # after training, save your model parameters in the dir 'saved_models'
    torch.save(model.state_dict(), model_dir + os.sep + model_name)