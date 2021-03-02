from typing import Tuple, Optional

import tensorflow as tf
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from tensorflow.keras import Model


def get_model(input_shape: Tuple[int, int, int],
              tuning: bool,
              l1_l2reg: float, 
              output_bias: Optional[float] = None):
    '''Returns a randomly initialized 2D convolutional model with transfer
    learning from VGG16, fixing the weights and replacing the classification layers
    with a set that ends in a single binary node.
    ::output_bias:: if you want to bootstrap the model for training on
    inbalanced data
    ::tuning:: if you want to mark the last convolutional block of VGG16 as
    trainable
    :: l1_l2reg :: regularization strength on classification layer'''

    if output_bias is not None:
      output_bias = tf.keras.initializers.Constant(output_bias)

    # adding the bottom layers from VGG16
    original = VGG16(weights='imagenet',
                     include_top=False,
                     # For global pooling at the ouput (alternative to Flatten)
                     #  pooling='avg',
                     input_shape=input_shape)
    
    # don't train the convolutional layers:
    for layer in original.layers[:]:
        layer.trainable = False
    
    # apply tuning configuration if needed
    if tuning:
      for layer in original.layers:
        if 'block5_conv' in layer.name:
          layer.trainable = True
    
    # connect to the last Max Pooling layer
    last = original.layers[-1].output

    # Flatten() or GlobalAveragePooling2D() is needed for connecting to a Dense layer
    # Flatten keeps spatial information which gave a bit better performance
    flat = Flatten()(last)

    dropped_a = Dropout(0.2)(flat)
    
    fully_connected_a = Dense(128,
                              kernel_regularizer = \
                              tf.keras.regularizers.l1_l2(l1 = l1_l2reg,
                                                          l2 = l1_l2reg),
                              activation='relu')(dropped_a)
    
    dropped_b = Dropout(0.2)(fully_connected_a)

    fully_connected_b = Dense(128,
                              kernel_regularizer = \
                              tf.keras.regularizers.l1_l2(l1 = l1_l2reg,
                                                          l2 = l1_l2reg),
                              activation='relu')(dropped_b)
    
    preds = Dense(1,
                  activation='sigmoid',
                  bias_initializer = output_bias)(fully_connected_b)
    
    model = Model(original.input, preds)
    
    return model