import os
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Activation,Dense
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.losses import categorical_crossentropy



train_dir = 'temple_images/temple_images/train'
test_dir = 'temple_images/temple_images/test'

# example_image = random.choice(os.listdir('temple_images/temple_images/train/golden'))
# example = plt.imread('temple_images/temple_images/train/golden/'+example_image)
# plt.imshow(example)
# plt.show()
# print(example.shape)


HEIGHT = 150
WIDTH = 150
EPOCHS = 22
BATCH_SIZE = 4
SAMPLE = 42

model = Sequential()

model.add(Conv2D(32,(3,3),input_shape=(WIDTH,WIDTH,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Conv2D(64,(3,3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# #
#
# model.add(Conv2D(128, (3,3),border_mode='same',activation='relu'))
# model.add(Conv2D(128, (3,3),border_mode='same',activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
#
# model.add(Dense(256,activation='relu'))
# model.add(Dropout(0.2))


# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))


model.add(Dense(7))
model.add(Activation('sigmoid'))

model.compile(loss='mean_squared_error',
   optimizer=Adam(),
   metrics=['accuracy'],
   )


train_data_gen = ImageDataGenerator(
       rescale = 1./255,
       shear_range = 0.2,
       zoom_range = 0.2,
       horizontal_flip = True,

)


training_generator = train_data_gen.flow_from_directory(
      train_dir,
      target_size = (WIDTH,HEIGHT),
      batch_size = BATCH_SIZE,
      class_mode = "categorical",
)

print('train',training_generator.class_indices)

model.fit_generator(
  training_generator,
  steps_per_epoch =SAMPLE // BATCH_SIZE,
  epochs = EPOCHS,
  verbose = 1,
)


model.save('model7_7_2.h5')
