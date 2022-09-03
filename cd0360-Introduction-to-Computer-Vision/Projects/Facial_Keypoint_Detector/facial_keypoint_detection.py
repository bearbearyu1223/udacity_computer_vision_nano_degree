import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import torch
from models import NaimishNet


if __name__ == "__main__":

    image = cv2.imread('images/obamas.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # # plot the image
    # fig = plt.figure(figsize=(9, 9))
    # plt.imshow(image)
    # plt.show()

    face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    image_with_detections = image.copy()

    # loop over the detected faces, mark the image where each face is found
    for (x, y, w, h) in faces:
        cv2.rectangle(image_with_detections, (x, y), (x + w, y + h), (255, 0, 0), 3)

    fig = plt.figure(figsize=(9, 9))
    plt.imshow(image_with_detections)
    plt.show()

    model = NaimishNet()
    model.load_state_dict(torch.load("saved_models/checkpoint.pt"))
    model.eval()

    image_copy = np.copy(image)

    # loop over the detected faces from your haar cascade
    i = 0
    fig = plt.figure(figsize=(10, 10))
    margin = int(w * 0.5)

    for (x, y, w, h) in faces:

        # Select the region of interest that is the face in the image
        roi = image_copy[max(y - margin, 0):min(y + h + margin, image.shape[0]), max(x - margin, 0):min(x + w + margin,
                                                                                                        image.shape[1])]

        ## TODO: Convert the face region from RGB to grayscale
        roi_copy = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        ## TODO: Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
        roi_copy = roi_copy / 255.0

        ## TODO: Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
        roi_copy = cv2.resize(roi_copy, (224, 224))
        if (len(roi_copy.shape) == 2):
            # add that third color dim
            roi_copy = roi_copy.reshape(roi_copy.shape[0], roi_copy.shape[1], 1)

        ## TODO: Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
        roi_copy = roi_copy.transpose((2, 0, 1))
        roi_copy = torch.from_numpy(roi_copy)

        ## TODO: Make facial keypoint predictions using your loaded, trained network
        roi_copy = roi_copy.reshape(1, roi_copy.shape[0], roi_copy.shape[1], roi_copy.shape[2])
        roi_copy = roi_copy.type(torch.FloatTensor)
        output_pts = model(roi_copy)
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)
        predicted_key_pts = output_pts.data
        predicted_key_pts = predicted_key_pts.numpy()
        predicted_key_pts = predicted_key_pts * 50.0 + 100
        predicted_key_pts = np.squeeze(predicted_key_pts)

        ## TODO: Display each detected face and the corresponding keypoints
        fig.add_subplot(1, len(faces), i + 1)
        plt.imshow(np.squeeze(roi_copy), cmap='gray')
        plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=5, marker='.', c='m')
        plt.axis('off')
        i += 1
    plt.show()


