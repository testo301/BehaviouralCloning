import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

# Defining generator for training/testing for memory optmization
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # In windows escape sign is needed for backslash
                # Captures forward/backward/curve path
                name = batch_sample[0].split('\\')[-3] + '/' + batch_sample[0].split('\\')[-2] + '/' +  batch_sample[0].split('\\')[-1]
                
                center_image = plt.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                # Augmenting the data
                images.append(cv2.flip(center_image,1))
                angles.append(center_angle*-1.0)
                
                # Left images processing by applying 0.25 additive scalar on the steering angle
                # Captures forward/backward/curve path
                left_name = batch_sample[1].split('\\')[-3] + '/' + batch_sample[1].split('\\')[-2] + '/' +  batch_sample[1].split('\\')[-1]
                left_image = plt.imread(left_name)
                left_angle = float(batch_sample[3]) + 0.25
                images.append(left_image)
                angles.append(left_angle )

                # Right images processing by applying 0.25 additive scalar on the steering angle
                # Captures forward/backward/curve path
                right_name = batch_sample[2].split('\\')[-3] + '/' + batch_sample[2].split('\\')[-2] + '/' +  batch_sample[2].split('\\')[-1]
   
                right_image = plt.imread(right_name)
                right_angle = float(batch_sample[3]) - 0.25
                images.append(right_image)
                angles.append(right_angle)
                # Augmenting the data
                #images.append(cv2.flip(center_image,1))
                #angles.append(center_angle*-1.0)
                
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Opening the log files and appending the paths to the data and the steering angles
samples = []
with open('forward/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
with open('backward/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
with open('curve/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
print('Sample size')
print(len(samples))        

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)




# Defining the model

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, ELU
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.models import Model
from keras.layers.convolutional import Convolution2D
from keras.callbacks import ModelCheckpoint

model = Sequential()

# Lambda layer
# Normalizing the inputs
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# Cropping the images to remove the unnecessary information.
# Cropping 70 pixels from the top of the image and cropping 25 pixels from the bottom of the image
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same",activation='elu'))
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same",activation='elu'))
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))

# Printing model summary
model.summary()

# Model compilation
model.compile(loss='mse', optimizer='adam')

# Establishing checkpoint rules
checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', monitor='val_loss', verbose=1, save_best_only=False, mode='auto')

# Model fitting and saving the object to plot model performance over epochs
history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=3, 
    callbacks=[checkpoint],                                 
    verbose=1)

# Printing the loss across the epochs

# Printing the keys
print(history_object.history.keys())
fig = plt.figure(figsize=(6, 6))
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='lewer right')
plt.show()
fig.savefig('images/lossplot.jpg')
plt.close(fig)




