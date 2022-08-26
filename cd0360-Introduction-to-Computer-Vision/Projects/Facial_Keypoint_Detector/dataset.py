import cv2
import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


def show_keypoints(image, key_pts):
    plt.imshow(image)
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker=".", c="m")
    plt.show()


class FacialKeypointsDataset(Dataset):
    def __init__(self, csv_file_path, image_dir, transform=None):
        self.key_pts_frame = pd.read_csv(csv_file_path, engine="python")
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.image_dir, self.key_pts_frame.iloc[idx, 0])
        image = mpimg.imread(image_name)
        if image.shape[2] == 4:
            image = image[:, :, 0:3]
        key_pts = self.key_pts_frame.iloc[idx, 1:].values
        key_pts = key_pts.astype("float").reshape(-1, 2)
        sample = {"image": image, "keypoints": key_pts}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]"""
    def __call__(self, sample):
        image, key_pts = sample["image"], sample["keypoints"]

        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image_copy = image_copy/255.0

        key_pts_copy = (key_pts_copy - 100.0) / 50.0

        return {"image": image_copy, "keypoints": key_pts_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size"""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample["image"], sample["keypoints"]
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size*h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size*w/h
        else:
            new_h, new_w = self.output_size

        img = cv2.resize(image, (int(new_w), int(new_h)))

        key_pts = key_pts * [new_w/w, new_h/h]
        return {"image": img, "keypoints": key_pts}


class RandomCrop(object):
    """ Crop Randomly the image in a sample"""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample["image"], sample["keypoints"]
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        key_pts = key_pts - [left, top]

        return {"image": image, "keypoints": key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors"""
    def __call__(self, sample):
        image, key_pts = sample["image"], sample["keypoints"]
        if len(image.shape) == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)
        # swap color axis because
        # numpy image: H X W X C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {"image": torch.from_numpy(image), "keypoints": torch.from_numpy(key_pts)}


def test():
    root_dir = os.path.join(os.getcwd(), "data")
    csv_file_name = "training_frames_keypoints.csv"
    csv_file_path = os.path.join(root_dir, csv_file_name)
    image_dir = os.path.join(root_dir, "training/")
    face_dataset = FacialKeypointsDataset(csv_file_path=csv_file_path, image_dir=image_dir)
    print("length of dataset: {}".format(len(face_dataset)))

    rescale = Rescale(100)
    crop = RandomCrop(50)
    composed = transforms.Compose([Rescale(250), RandomCrop(224)])

    test_num = 500
    sample = face_dataset[test_num]

    fig = plt.figure()
    for i, tx in enumerate([rescale, crop, composed]):
        transformed_sample = tx(sample)

        ax = plt.subplot(1, 3, i + 1)
        plt.tight_layout()
        ax.set_title(type(tx).__name__)
        show_keypoints(transformed_sample["image"], transformed_sample["keypoints"])
    plt.show()


if __name__ == "__main__":
    data_transform = transforms.Compose([Rescale(250), RandomCrop(224), Normalize(), ToTensor()])
    root_dir = os.path.join(os.getcwd(), "data")
    csv_file_name = "training_frames_keypoints.csv"
    csv_file_path = os.path.join(root_dir, csv_file_name)
    image_dir = os.path.join(root_dir, "training/")
    transformed_dataset = FacialKeypointsDataset(csv_file_path=csv_file_path, image_dir=image_dir, transform=data_transform)

    print("Number of Images: {}".format(len(transformed_dataset)))
    for i in range(5):
        sample = transformed_dataset[i]
        print(i, sample['image'].size(), sample['keypoints'].size())



