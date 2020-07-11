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

XAuthenticate = list(np.load(config.np_casia_two_au_path))
yAuthenticate = list(np.zeros(len(XAuthenticate), dtype=np.uint8))
XForged = list(np.load(config.np_casia_two_forged_path))
yForged = list(np.ones(len(XForged), dtype=np.uint8))

X = np.array(XAuthenticate + XForged)
y = np.array(yAuthenticate + yForged, dtype=np.int8)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True)

plt.hist(y, bins=5)
plt.ylabel("Number of images")
plt.title("CASIA II - Authenticate OR Fake Image Dataset")
plt.show()
img_height = 256
img_width = 384

# build the VGG16 network
model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))
# add the model on top of the convolutional base

# model.add(top_model) this throws error alternative is below

new_model = Sequential() #new model
for layer in model.layers:
    new_model.add(layer)

new_model.add(top_model) # now this works

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
# LOCK THE TOP CONV LAYERS
for layer in new_model.layers[:15]:
    layer.trainable = False

print('Model loaded.')

print(new_model.summary())

# model_aug.load_weights('k64 binary 25percent stride8/fine_tuned_model_resnet_64_adam_weights.h5')
# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
new_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

new_model.fit(x_train, y_train,
          epochs=100,
          batch_size=10)

new_model.evaluate(x_test, y_test, verbose=0)

y_pred = new_model.predict_classes(x_test)

new_model.save(filepath='casia2_model.h5')
# new_model.save_weights('bottleneck_fc_model.h5')
plt.clf()
print(confusion_matrix(y_test, y_pred))
