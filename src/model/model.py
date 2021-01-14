from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D
from keras.models import Sequential
from keras.optimizers import Adam
import requests

def mole_model():
 
    model = Sequential()
    model.add(Conv2D(32, (5, 5), strides=(1, 1), padding='same', input_shape=(256,256,3), activation='relu'))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', activation='relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu')) # Layer 4
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model 
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

