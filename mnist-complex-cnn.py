import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('tf')  # using tensorflow format

'''
This model uses a second layer convolutional layer, and a second fully connected layer.
It gets a slightly better accuracy than the simple cnn.

'''

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, Y_train), (X_test,Y_test) = mnist.load_data()
# reshape to be [samples][width[height][channel]
# each image is 28 x 28 pixels, 1 channel for grayscale
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# normalize inputs from 0-255 (pixel intensity) to 0-1
X_train = X_train / 255
X_test = X_test / 255
# conver to one hot encoding
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]

def baseline_model():
    '''
    
    This function function initializes the CNN architecture, and returns the model. Note,
    Keras automatically matches the output layer to the input of the next layer (unlike
    in plain TensorFlow).
    
    '''
    # create model
    model = Sequential()
    model.add(Conv2D(32, (5,5), input_shape=(28,28,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(16, (3,3), input_shape=(28,28,1), activation='relu'))  
    model.add(Dropout(0.2))  # for regularization
    model.add(Flatten())  # need to do this before the FC layer
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
# build the model
model = baseline_model()
# Fit the model, or train the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

















    
