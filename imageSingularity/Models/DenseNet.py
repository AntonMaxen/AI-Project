from imageSingularity.Models import Model
import tensorflow

import pandas as pd
import numpy as np
import os
import keras
import random
import cv2
import math
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Convolution2D,
    BatchNormalization, Flatten, MaxPooling2D, Dropout
)

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import warnings
warnings.filterwarnings("ignore")


class DenseNet(Model):
    def __init__(self, images):
        super(DenseNet).__init__(images)
        self.categories = images.keys()
        self.base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=images[self.categories[0]]['images'][0].shape
        )

        self.x = self.base_model.output
        self.x = GlobalAveragePooling2D()(self.x)
        self.x = BatchNormalization()(self.x)
        self.x = Dropout(0.5)(self.x)
        self.x = Dense(1024, activation='relu')(self.x)
        self.x = Dense(512, activation='relu')(self.x)
        self.x = BatchNormalization()(self.x)
        self.x = Dropout(0.5)(self.x)

        self.predictions = Dense(len(self.categories), activation='softmax')(self.x)


if __name__ == '__main__':
    pass
