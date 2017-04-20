"""
This file reads the file names and angles for several images from several different manual runs in the simulator.

It does also define the generators used when training and validating the neural network.
"""

import os
import csv
import sklearn
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import config

"""Code for reading the images and csv for the different runs.
They are saved in different folder in the following structure in order to allow me to
easily remove certain runs should it be needed.

../drivingData
    /easy
        /run1
        /run2
        /run3
    /hard
        /run1
        /run2
        /run3

The choice of reading from the easy and/or hard catalog is set in a dict in config.py
"""

relativePathDrivingData = "../drivingData"
images = []

for difficulty in config.difficulty:
    if config.difficulty[difficulty]:
        difficultyRelativePath = relativePathDrivingData + "/" + difficulty
        listCatalogsInDifficulty = os.listdir(difficultyRelativePath)

        for catalog in listCatalogsInDifficulty:
            relativePath = difficultyRelativePath + "/" + catalog

            with open(relativePath + "/driving_log.csv") as drivingLog:
                reader = csv.reader(drivingLog)
                for i, line in enumerate(reader):
                    center = line[0]
                    left = line[1]
                    right = line[2]
                    angle = float(line[3])

                    """Storing center, left, right and flipped versions of them separately in order
                    to draw between them independently while making training / validation batches."""
                    images.append({
                        'image': center,
                        'angle': angle,
                        'flipped': False})

                    images.append({
                        'image': left,
                        'angle': angle + config.corretionAngle,
                        'flipped': False})

                    images.append({
                        'image': right,
                        'angle': angle - config.corretionAngle,
                        'flipped': False})

                    images.append({
                        'image': center,
                        'angle': -angle,
                        'flipped': True})

                    images.append({
                        'image': left,
                        'angle': -(angle + config.corretionAngle),
                        'flipped': True})

                    images.append({
                        'image': right,
                        'angle': -(angle - config.corretionAngle),
                        'flipped': True})

def generator(imageInput, batch_size = 32):
    num_samples = len(imageInput)
    while 1:
        # Shuffle between each epoch
        imageInput = sklearn.utils.shuffle(imageInput)

        for offset in range(0, num_samples, batch_size):
            batch_images = imageInput[offset:offset + batch_size]
            images = []
            angles = []
            for image in batch_images:
                imageLoaded = cv2.imread(image['image'])
                if image['flipped']:
                    imageLoaded = cv2.flip(imageLoaded, 1)

                images.append(imageLoaded)
                angles.append(image['angle'])

            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)

train_images, validation_images = train_test_split(images,
                                                   test_size=config.portionOfImagesForValidation)

trainingGenerator = generator(train_images, batch_size=config.batchSize)
validationGenerator = generator(validation_images, batch_size=config.batchSize)