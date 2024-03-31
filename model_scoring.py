
import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.applications.inception_v3 import InceptionV3
from scipy import *
from utils import *


def create_model(img_size):
    inputt1 = tensorflow.keras.layers.Input(shape=(img_size, img_size, 3))
    inputt2 = tensorflow.keras.layers.Input(shape=(img_size, img_size, 3))
    m = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    m.summary()
    x = m(inputt1)
    x2 = m(inputt2)
    x = tensorflow.keras.layers.Flatten()(x)
    x2 = tensorflow.keras.layers.Flatten()(x2)
    L1_layer = tensorflow.keras.layers.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([x, x2])
    prediction = tensorflow.keras.layers.Dense(1, activation='sigmoid')(L1_distance)
    modelo = tensorflow.keras.models.Model(inputs=[inputt1, inputt2], outputs=prediction)

    modelo.summary()
    modelo.compile(loss='binary_crossentropy',optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.0002, decay=1e-7), metrics=['accuracy'])
    return modelo