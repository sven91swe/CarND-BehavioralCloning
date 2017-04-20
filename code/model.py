"""
This file defines the preprocessing of images and the neural network both implemented in Keras.

It also contains the code that trains and saves the trained neural network to a file.
"""


from keras.models import Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, Lambda, Cropping2D, Input, Concatenate
from keras.initializers import Constant

import config

dropOutRate = config.dropOutRate
activationFunctionConvolutional = "relu"
activationFunctionFullyConnected = "tanh"

"""
Input layer with the shape of the loaded color images"""
inLayer = Input(shape=(160,320,3))

"""
Cropping the image and adding a forth layer for grayscale, also some normalization."""
crop = Cropping2D(cropping=((70, 24), (0, 0)))(inLayer)
color = Lambda(lambda x: (x/255) - 0.5)(crop)
gray = Conv2D(1, 1, use_bias=False, kernel_initializer=Constant(1/3), trainable=False)(color)
colorAndGray = Concatenate()([color, gray])

"""
Network based on article published by Nvidia: 'End to End Learning for Self-Driving Cars'

Modifications:
I have added a forth input channel that is in grayscale in addition to the previous red, blue and green.
Should help training in the case that color information is less important than structure.

As I added an input channel I have also increased the number of features or neurons depending on layer
by 1-1.5 the number used by Nvidia.

The fully connected layers uses tanh instead of relu as their activation function.
Finally I have added dropout layers for training purposes in order to avoid overfitting.
"""

conv1 = Conv2D(36, (5, 5), strides=(2, 2), activation=activationFunctionConvolutional)(colorAndGray)
drop1 = Dropout(dropOutRate)(conv1)

conv2 = Conv2D(48, (5, 5), strides=(2, 2), activation=activationFunctionConvolutional)(drop1)
drop2 = Dropout(dropOutRate)(conv2)

conv3 = Conv2D(64, (5, 5), strides=(2, 2), activation=activationFunctionConvolutional)(drop2)
drop3 = Dropout(dropOutRate)(conv3)

conv4 = Conv2D(80, (3, 3), activation=activationFunctionConvolutional)(drop3)
drop4 = Dropout(dropOutRate)(conv4)

conv5 = Conv2D(80, (3, 3), activation=activationFunctionConvolutional)(drop4)
drop5 = Dropout(dropOutRate)(conv5)


flattened = Flatten()(drop5)
fullyConnected1 = Dense(150, activation=activationFunctionFullyConnected)(flattened)
dropFC1 = Dropout(dropOutRate)(fullyConnected1)

fullyConnected2 = Dense(75, activation=activationFunctionFullyConnected)(dropFC1)
dropFC2 = Dropout(dropOutRate)(fullyConnected2)

fullyConnected3 = Dense(15, activation=activationFunctionFullyConnected)(dropFC2)
outLayer = Dense(1)(fullyConnected3)

model = Model(inputs=inLayer, outputs= outLayer)

print(Model(inputs=inLayer, outputs=conv5).output_shape)
"""
I had this section in a separate file in order to have model.py only describing the model architecture.

But I interpret the requirement for this project in such a way so I need to place it here even though
I think the code that trains the network would fit better in a separate 'main' file.
"""
if __name__ == "__main__":
    import dataLoader

    trainingGenerator = dataLoader.trainingGenerator
    validationGenerator = dataLoader.validationGenerator

    model.compile(loss='mse', optimizer='adam')

    numberOfTrainingImages = len(dataLoader.train_images)
    numberOfValidationImages = len(dataLoader.validation_images)

    model.fit_generator(trainingGenerator,
                        steps_per_epoch=numberOfTrainingImages // config.batchSize + 1,
                        validation_data=validationGenerator,
                        validation_steps=numberOfValidationImages // config.batchSize + 1,
                        nb_epoch=config.numberOfEpochs,
                        verbose=2)

    print("Saving model")
    model.save('../model.h5')
