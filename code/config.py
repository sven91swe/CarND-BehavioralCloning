

"""
What track should be trained on?
As the data from different runs and tracks are stored in different folders,
 this setting control from which tracks the neural network shall train on.
"""
difficulty = {'easy': False,
              'hard': True}

#Training related parameters.
batchSize = 64
numberOfEpochs = 50
dropOutRate = 0.25

portionOfImagesForValidation = 0.2


#The difference in 'ground truth' angle between an image from the center camera to images from the left / right camera.
corretionAngle = 0.2