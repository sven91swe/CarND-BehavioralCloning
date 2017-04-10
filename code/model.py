from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, Lambda, Cropping2D

import dataLoader
import config

trainingGenerator = dataLoader.trainingGenerator
validationGenerator = dataLoader.validationGenerator

dropOutRate = config.dropOutRate
activationFunctionConvolutional = "relu"
activationFunctionFullyConnected = "tanh"

model = Sequential()
model.add(Cropping2D(cropping=((70, 24), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x/255) - 0.5))

model.add(Conv2D(36, (5, 5), strides=(2, 2), activation=activationFunctionConvolutional))
model.add(Dropout(dropOutRate))

model.add(Conv2D(48, (5, 5), strides=(2, 2), activation=activationFunctionConvolutional))
model.add(Dropout(dropOutRate))

model.add(Conv2D(64, (5, 5), strides=(2, 2), activation=activationFunctionConvolutional))
model.add(Dropout(dropOutRate))

model.add(Conv2D(80, (3, 3), activation=activationFunctionConvolutional))
model.add(Dropout(dropOutRate))

model.add(Conv2D(80, (3, 3), activation=activationFunctionConvolutional))
model.add(Dropout(dropOutRate))


model.add(Flatten())
model.add(Dense(150, activation=activationFunctionFullyConnected))
model.add(Dropout(dropOutRate))

model.add(Dense(75, activation=activationFunctionFullyConnected))
model.add(Dropout(dropOutRate))

model.add(Dense(15, activation=activationFunctionFullyConnected))
model.add(Dense(1))

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
