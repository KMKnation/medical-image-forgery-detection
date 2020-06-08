#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf

import os
import cv2
import gc
from skimage import color, data, restoration
import cv2
import numpy as np
from skimage.restoration import estimate_sigma
from skimage.filters import median
import config
import imutils
import warnings
warnings.filterwarnings('ignore')


# In[24]:


def weiner_noise_reduction(img):
    # data.astronaut()
    img = color.rgb2gray(img)
    from scipy.signal import convolve2d
    psf = np.ones((5, 5)) / 25
    img = convolve2d(img, psf, 'same')
    img += 0.1 * img.std() * np.random.standard_normal(img.shape)
    deconvolved_img = restoration.wiener(img, psf, 1100)

    return deconvolved_img



def estimate_noise(img):
    # img = cv2.imread(image_path)
    return estimate_sigma(img, multichannel=True, average_sigmas=True)


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    enoise = estimate_noise(image)
    noise_free_image = weiner_noise_reduction(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    fingerprint = gray - noise_free_image
    fingerprint = fingerprint / 255
    filtered_img = median(fingerprint, selem=None, out=None, mask=None, shift_x=False,
                          shift_y=False, mode='nearest', cval=0.0, behavior='rank')
    colored = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
    # print('-----------------')
    # cv2.imshow('filtered_image', filtered_img)
    # colored = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
    # print(colored)
    # cv2.imshow('colored', colored)
    return colored


# In[25]:


import keras
from keras import Model, Sequential, optimizers, applications
from keras.applications import ResNet50
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, Flatten
from keras_applications import resnet50
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import config


# In[26]:


DDSM_DATASET = 'CDDSM/figment.csee.usf.edu/pub/DDSM/cases'


# In[27]:


NORMAL = os.path.join(DDSM_DATASET, 'normals')
ABNORMAL = os.path.join(DDSM_DATASET, 'cancers')


# In[28]:


import glob
import random
normalGlob = glob.glob(NORMAL+"/*/*/*.jpg")
abNormalGlob = glob.glob(ABNORMAL+"/*/*/*.jpg")


# In[29]:


print(len(normalGlob))


# In[30]:


print(len(abNormalGlob))


# In[31]:


normalGlob[:2]


# In[32]:


def data_generator(normalGlob, abNormalGlob, BATCH_SIZE):
    while True:
        images = []
        labels = []
        img_height = 256
        img_width = 384
        random.shuffle(normalGlob)
        random.shuffle(abNormalGlob)

        if BATCH_SIZE == None:
            BATCH_SIZE = 32

        NORMAL_RATIO = int(BATCH_SIZE / 2)
        ABNORMAL_RATIO = int(BATCH_SIZE - NORMAL_RATIO)

        for imagepath in normalGlob[:NORMAL_RATIO]:
            image = cv2.imread(imagepath)
            image = cv2.resize(image, (img_width, img_height))
            image = preprocess_image(image)
#             image = image / 255
            images.append(image)
            labels.append(0)

        for imagepath in normalGlob[:ABNORMAL_RATIO]:
            image = cv2.imread(imagepath)
            image = cv2.resize(image, (img_width, img_height))
            image = preprocess_image(image)
#             image = image / 255
            images.append(image)
            labels.append(1)

        temp = list(zip(images, labels)) 
        random.shuffle(temp) 
        images, labels = zip(*temp)
    #     print(np.array(images).shape)
    #     print(np.array(labels).shape)

        yield np.array(images), np.array(labels)


# In[33]:


data_generator(normalGlob, abNormalGlob, 2)


# In[34]:


img_height = 256
img_width = 384

# build the VGG16 network
model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))


# In[11]:


top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))


# In[12]:


# model.add(top_model) this throws error alternative is below

new_model = Sequential() #new model
for layer in model.layers:
    new_model.add(layer)

new_model.add(top_model) # now this works


# In[13]:


# model_aug.load_weights('99 % accurate model.h5')
# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
new_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


# In[18]:


from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import LearningRateScheduler
batch_size = 32
num_epochs = 100
# input_shape = (224, 224, 3)
validation_split = .2
verbose = 1
patience = 50

def scheduler(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.001 * (0.1 ** int(epoch/10))
checkpoint = ModelCheckpoint(filepath='vgg.h5',
                             monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto',
                             period=2)

early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
callback = LearningRateScheduler(scheduler)


# In[ ]:


hist = new_model.fit_generator(steps_per_epoch=num_epochs // batch_size,generator=data_generator(normalGlob, abNormalGlob, batch_size)
                           , validation_data=data_generator(normalGlob, abNormalGlob, 12)
                           , validation_steps=num_epochs // batch_size,epochs=num_epochs,callbacks=[callback, checkpoint, early])


# In[ ]:




