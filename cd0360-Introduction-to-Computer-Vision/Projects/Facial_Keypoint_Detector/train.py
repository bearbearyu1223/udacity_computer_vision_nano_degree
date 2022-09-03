import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataset import FacialKeypointsDataset, Rescale, Normalize, ToTensor
from models import NaimishNet
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, utils
from torch.optim import lr_scheduler


def create_transformed_train_test_val_dataset(train_csv_file_path, train_image_dir,
                                              test_csv_file_path, test_image_dir, transform, valid_size=0.2):
    train_dataset = FacialKeypointsDataset(csv_file_path=train_csv_file_path, image_dir=train_image_dir,
                                           transform=transform)
    test_dataset = FacialKeypointsDataset(csv_file_path=test_csv_file_path, image_dir=test_image_dir,
                                          transform=transform)
    num_test = len(test_dataset)
    indices = list(range(num_test))
    np.random.shuffle(indices)
    split = int(valid_size * num_test)
    test_idx, valid_idx = indices[split:], indices[:split]

    test_sampler = SubsetRandomSampler(test_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    return train_dataset, test_dataset, test_sampler, valid_sampler


def create_train_test_val_dataloader(train_dataset, test_dataset, batch_size, test_sampler, valid_sampler,
                                     shuffle=True, num_workers=0):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)
    valid_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
    return train_loader, test_loader, valid_loader


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


def validation_loss(valid_loader, model, criterion = nn.SmoothL1Loss()):
    model.eval()
    loss = 0.0
    running_loss = 0.0
    for i, batch in enumerate(valid_loader):
        images = batch["image"]
        key_pts = batch["keypoints"]
        key_pts = key_pts.view(key_pts.size(0), -1)
        key_pts = key_pts.type(torch.FloatTensor)
        images = images.type(torch.FloatTensor)

        output_pts = model(images)
        loss = criterion(output_pts, key_pts)
        running_loss += loss.item()
    avg_loss = running_loss / (i + 1)
    model.train()
    return avg_loss


class EarlyStopping:
    def __init__(self, patience=15):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        print("Validation loss decreased from {0:.6f} to {0:.6f}, saving model ...".format(self.val_loss_min, val_loss))
        model_dir = os.path.join(os.getcwd(), 'saved_models')
        model_name = 'checkpoint.pt'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        # after training, save your model parameters in the dir 'saved_models'
        torch.save(model.state_dict(), model_dir + os.sep + model_name)
        self.val_loss_min = val_loss


def train(model, n_epochs, train_loader, valid_loader, scheduler, optimizer, criterion=nn.SmoothL1Loss()):
    train_loss_overtime = []
    valid_loss_overtime = []
    early_stopping = EarlyStopping()
    model.train()

    for epoch in range(n_epochs):

        running_train_loss = 0.0
        avg_val_loss = 0.0
        avg_train_loss = 0.0

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

            running_train_loss += loss.item()
            if batch_i % 10 == 9:  # print every 10 batches
                avg_val_loss = validation_loss(valid_loader, model)
                train_loss_overtime.append(avg_train_loss)
                valid_loss_overtime.append(avg_val_loss)
                print('Epoch: {}, Batch: {}, Avg. Train Loss: {}, Avg. Validation Loss: {}'.format(epoch + 1, batch_i + 1, running_train_loss / 10, avg_val_loss))
                running_train_loss = 0.0
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            train_loss_over_time = train_loss_overtime[:-early_stopping.patience]
            val_loss_over_time = valid_loss_overtime[:-early_stopping.patience]
            print("Early stopping")
            break
    model.load_state_dict(torch.load('saved_models/checkpoint.pt'))
    print('Finished Training')
    return train_loss_overtime, valid_loss_overtime, epoch + 1


if __name__ == "__main__":
    root_dir = os.path.join(os.getcwd(), "data")
    train_csv_file_name = "training_frames_keypoints.csv"
    test_csv_file_name = "test_frames_keypoints.csv"
    train_csv_file_path = os.path.join(root_dir, train_csv_file_name)
    test_csv_file_path = os.path.join(root_dir, test_csv_file_name)
    train_image_dir = os.path.join(root_dir, "training/")
    test_image_dir = os.path.join(root_dir, "test/")
    transform = transforms.Compose([Rescale((224, 224)), Normalize(), ToTensor()])
    train_dataset, test_dataset, test_sampler, valid_sampler = create_transformed_train_test_val_dataset(
        train_csv_file_path, train_image_dir, test_csv_file_path, test_image_dir, transform)
    batch_size = 16
    train_loader, test_loader, valid_loader = create_train_test_val_dataloader(train_dataset, test_dataset, batch_size,
    test_sampler, valid_sampler)
    model = NaimishNet()

    n_epochs = 10  # start small, and increase when you've decided on your model structure and hyperparams

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, verbose=True)
    train_loss_over_time, val_loss_over_time, epoch = train(model, n_epochs, train_loader, valid_loader, scheduler, optimizer)
    test_images, test_outputs, gt_pts = test_net_sample_output(test_loader, model)
    visualize_output(test_images, test_outputs, gt_pts)

