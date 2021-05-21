import os
from imageSingularity.image import Images
from imageSingularity.utils import create_image_db, get_project_root
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np

from tensorflow import keras
from imageSingularity.Models import IsModel
from tensorflow.keras.layers import Dense

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model


import warnings
warnings.filterwarnings("ignore")


def reformat_images(image_dict):
    all_images = []
    all_labels = []
    category_lengths = [len(v['images']) for _, v in image_dict.items()]
    min_cat_len = min(category_lengths)

    for k, v in image_dict.items():
        image_list = v['images']
        all_images.extend([t_image for i, t_image in enumerate(image_list) if i < min_cat_len])
        all_labels.extend([k for i, _ in enumerate(image_list) if i < min_cat_len])

    all_images = np.array(all_images)
    print(len(all_images))
    all_labels = np.array(all_labels)
    print(len(all_labels))

    return all_images, all_labels


class DenseNet(IsModel):
    def __init__(self):
        super(DenseNet).__init__()
        self.categories = None
        self.amount_categories = 0
        self.base_model = DenseNet121(
            weights='imagenet',
            include_top=True
        )
        self.model = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.x_val = None
        self.y_val = None

    def load_data(self, np_images):
        self.images = np_images
        self.categories = list(self.images.keys())
        self.amount_categories = len(self.categories)
        # Some stuff
        all_images, all_labels = reformat_images(self.images)
        mlb = LabelBinarizer()
        all_labels = mlb.fit_transform(all_labels)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            all_images,
            all_labels,
            test_size=0.4,
            random_state=42
        )

    def setup(self):
        x = self.base_model.output
        x = Dense(self.amount_categories, activation='softmax')(x)
        self.model = Model(inputs=self.base_model.input, outputs=x)

        for layer in self.model.layers[:-8]:
            layer.trainable = False

        for layer in self.model.layers[-8:]:
            layer.trainable = True

        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, checkpoint_list, epochs=5):
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs,
            verbose=2,
            callbacks=checkpoint_list,
            validation_data=(self.x_train, self.y_train)
        )

    def evalute(self):
        loss, accuracy = self.model.evaluate(
            self.x_test,
            self.y_test,
            len(self.y_test)
        )

        return loss, accuracy

    def predict(self, test_values):
        if not isinstance(test_values, list):
            test_values = [test_values]

        return self.model.predict(test_values)

    def load(self, path):
        self.model = keras.models.load_model(path)


def create_db():
    base = get_project_root()
    db_path = os.path.join(base, "images.hdf5")
    dataset_path = os.path.join(base, "wikiarta")
    categories = os.listdir(dataset_path)
    create_image_db(100000000, dataset_path, categories, (224, 224), db_path)


def test_model_load():
    model = keras.models.load_model('model.h5')
    model.evalutate()


def dense_net_test():
    db_path = os.path.join(get_project_root(), 'images.hdf5')
    images = Images()
    categories = ['Rococo', 'Color_Field_Painting', 'Art_Nouveau_Modern']
    images.load(db_path, categories)

    dn = DenseNet()
    # Load images and create training and split data.
    dn.load_data(images.images)
    # Creates Model
    dn.setup()
    anne = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-3)
    checkpoint = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
    checkpoint_list = [anne, checkpoint]
    # train class
    dn.train(checkpoint_list, epochs=2)
    print(dn.evalute())