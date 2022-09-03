import torch
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
from train import create_transformed_train_test_val_dataset, create_train_test_val_dataloader, visualize_output, test_net_sample_output

if __name__ == "__main__":
    root_dir = os.path.join(os.getcwd(), "data")
    train_csv_file_name = "training_frames_keypoints.csv"
    test_csv_file_name = "test_frames_keypoints.csv"
    train_csv_file_path = os.path.join(root_dir, train_csv_file_name)
    test_csv_file_path = os.path.join(root_dir, test_csv_file_name)
    train_image_dir = os.path.join(root_dir, "training/")
    test_image_dir = os.path.join(root_dir, "test/")
    transform = transforms.Compose([Rescale((224, 224)), Normalize(), ToTensor()])
    train_dataset, test_dataset = create_transformed_train_test_val_dataset(train_csv_file_path, train_image_dir,
                                                                            test_csv_file_path, test_image_dir, transform,
                                                                            verbose=False)
    train_loader, test_loader = create_train_test_val_dataloader(train_dataset, test_dataset,
                                                                 batch_size=10, shuffle=True, num_workers=0)

    model = NaimishNet()
    model_dir = 'saved_models/'
    model_name = 'keypoints_model_1.pt'
    model.load_state_dict(torch.load(model_dir + model_name))
    model.eval()

    test_images, test_outputs, gt_pts = test_net_sample_output(test_loader, model)
    visualize_output(test_images, test_outputs, gt_pts)


