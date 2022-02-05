# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras import regularizers


class DefoNet:
    @staticmethod
    def build(width, height, depth, classes, finalAct="softmax"):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        l = 0.001
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(32, (3, 3), padding="same",
                          input_shape=inputShape, kernel_regularizer=regularizers.l2(l)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same",
                         input_shape=inputShape, kernel_regularizer=regularizers.l2(l)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        # (CONV => RELU) * 3 => POOL
        model.add(Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        # (CONV => RELU) * 3 => POOL
        model.add(Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        # model.add(GlobalAveragePooling2D())
        model.add(Dense(1024, kernel_regularizer=regularizers.l2(l)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.35))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation(finalAct))
        # return the constructed network architecture
        return model
