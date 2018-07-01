import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import matplotlib.pyplot as plt

'''
Baseline MNIST model with Multi-Layer Perceptrons (fully connected network).

'''


# to ensure same random numbers generated each time
seed = 7
np.random.seed(seed)

# load (downloaded if needed) the MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# flatten 28*28 images to a 784 vector for each images
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# convert labels to one hot style, binary
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
# build the model
model = baseline_model()
# fit the model
model.fit(X_train, Y_train, validation_data=(X_test,Y_test), epochs=10, batch_size=200, verbose=2)
# final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Baseline error: %.2f" % (100-scores[1]*100))
