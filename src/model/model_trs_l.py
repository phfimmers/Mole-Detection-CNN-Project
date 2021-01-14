from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D
from keras.models import Sequential
from keras.optimizers import Adam
import requests

def mole_model():
 
    model = Sequential()
    # adding the input layer to vgg model
    vgg = VGG16(weights='imagenet',include_top=False, input_shape=(64,64,3))
    model.add(vgg)

    model.add(Flatten())
    model.add(Dense(2, activation="softmax"))
    
    # don't train the vgg layers:
    for layers in model.layers[0].layers:
        layers.trainable = False

    # And Freeze also layer zero for good measure
    model.layers[0].trainable = False

    
    # Compile model 
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model