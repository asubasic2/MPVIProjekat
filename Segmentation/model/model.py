import keras
from keras.models import *
from keras.layers import *
from keras_segmentation.models.model_utils import get_segmentation_model


class Cnn:
    @staticmethod
    def build(width, height, depth, classes):
        img_input = Input(shape=(height, width, depth))

        conv1 = Conv2D(8, (5, 5), padding="same", activation='relu')(img_input)
        conv1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(16, (3, 3), activation='relu', padding="same")(pool1)
        conv2 = Dropout(0.5)(conv2)
        conv2 = Conv2D(16, (3, 3), activation='relu', padding="same")(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(32, (3, 3), activation='relu', padding="same")(pool2)
        conv3 = Dropout(0.5)(conv3)
        conv3 = Conv2D(32, (3, 3), activation='relu', padding="same")(conv3)

        up1 = concatenate([UpSampling2D((2, 2))(conv3), conv2], axis=-1)
        conv4 = Conv2D(16, (3, 3), activation='relu', padding='same')(up1)
        conv4 = Dropout(0.5)(conv4)
        conv4 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv4)

        up2 = concatenate([UpSampling2D((2, 2))(conv4), conv1], axis=-1)
        conv5 = Conv2D(8, (3, 3), activation='relu', padding='same')(up2)
        conv5 = Dropout(0.5)(conv5)
        conv5 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv5)

        out = Conv2D(classes, (1, 1), padding='same')(conv5)
        model = get_segmentation_model(img_input, out)

        return model
