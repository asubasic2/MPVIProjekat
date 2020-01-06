import keras
from keras.models import *
from keras.layers import *
from keras.models import Model
from keras_segmentation.models.model_utils import get_segmentation_model


class CNN:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        img_input = Input(shape=(height, width, depth))
        chanDim = -1

        conv1 = Conv2D(8, (5, 5), padding="same", activation='relu')(img_input)
        conv1 = BatchNormalization(axis=chanDim)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(16, (3, 3), padding="same", activation='relu')(pool1)
        conv2 = BatchNormalization(axis=chanDim)(conv2)
        conv2 = Conv2D(16, (3, 3), padding="same", activation='relu')(conv2)
        conv2 = BatchNormalization(axis=chanDim)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(32, (3, 3), padding="same", activation='relu')(pool2)
        conv3 = BatchNormalization(axis=chanDim)(conv3)
        conv3 = Conv2D(32, (3, 3), padding="same", activation='relu')(conv3)
        conv3 = BatchNormalization(axis=chanDim)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        flat1 = Flatten()(pool3)
        flat1 = Dense(128)(flat1)
        flat1 = Activation("relu")(flat1)
        flat1 = BatchNormalization(axis=chanDim)(flat1)
        flat1 = Dropout(0.5)(flat1)

        out = Dense(classes)(flat1)
        out = Activation("softmax")(out)

        model = Model(inputs=img_input, outputs=out)

        return model
