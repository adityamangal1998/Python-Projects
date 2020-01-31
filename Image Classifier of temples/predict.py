from os import listdir
from os.path import isfile,join
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing import image
import matplotlib.pyplot as plt


model = load_model('model7_7_2.h5')
model.compile(loss='mean_squared_error',
               optimizer='rmsprop',
               metrics=['accuracy'])

test_dir = 'temple_images/temple_images/test/'

onlyfiles = [f for f in listdir(test_dir) if isfile(join(test_dir, f))]
print(onlyfiles)

for files in onlyfiles:
    img = image.load_img(test_dir+files,target_size=(150,150))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)

    images = np.vstack([x])
    # plt.imshow(images)
    classes = model.predict_classes(images,batch_size=3)
    print(classes)
    # classes = classes[0]
