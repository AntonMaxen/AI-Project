import os
import pickle

import tensorflow.python.framework.ops
import tensorflow as tf

from imageSingularity.image import Images
from imageSingularity.utils import create_image_db, get_project_root
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix

import numpy as np

from tensorflow import keras
from imageSingularity.Models import IsModel
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D,
    BatchNormalization, Dropout
)

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model

import warnings
warnings.filterwarnings("ignore")

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'TRUE'
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tensorflow.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

    except RuntimeError as e:
        print(e)


def save_data(data, filename):
    with open(filename, 'wb+') as p_file:
        pickle.dump(data, p_file)


def reformat_images(image_dict, cat_len=500):
    all_images = []
    all_labels = []
    category_lengths = [len(v['images']) for _, v in image_dict.items()]
    min_cat_len = min(category_lengths)

    cat_len = min_cat_len if min_cat_len < cat_len else cat_len

    for k, v in image_dict.items():
        image_list = v['images']
        all_images.extend([t_image for i, t_image in enumerate(image_list) if i < cat_len])
        all_labels.extend([k for i, _ in enumerate(image_list) if i < cat_len])

    all_images = np.array(all_images)
    print(len(all_images))
    all_labels = np.array(all_labels)
    print(len(all_labels))

    return all_images, all_labels


class DenseNet(IsModel):
    def __init__(self, shape=(224, 224, 3)):
        super(DenseNet).__init__()
        self.categories = None
        self.amount_categories = 0
        self.base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=shape
        )
        self.model = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.x_val = None
        self.y_val = None

    def load_data(self, np_images, cat_len=500):
        self.images = np_images
        self.categories = list(self.images.keys())
        self.amount_categories = len(self.categories)
        # Some stuff
        all_images, all_labels = reformat_images(self.images, cat_len=cat_len)
        all_images = np.array(all_images, dtype="float32") / 255.0
        mlb = LabelBinarizer()
        all_labels = mlb.fit_transform(all_labels)

        if self.amount_categories < 3:
            all_labels = tensorflow.keras.utils.to_categorical(all_labels)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            all_images,
            all_labels,
            test_size=0.2,
            random_state=42
        )

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x_train,
            self.y_train,
            test_size=0.2,
            random_state=42
        )

        test_data = (self.x_test, self.y_test)
        save_data(test_data, 'test_data.pickle')

    def setup(self):
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(self.amount_categories, activation='softmax')(x) # Classification layer
        self.model = Model(inputs=self.base_model.input, outputs=x)

        """
        Leave the base model alone and just train last layers and your fully connected classification layer.
        """
        for layer in self.model.layers[:-8]:
            layer.trainable = False

        for layer in self.model.layers[-8:]:
            layer.trainable = True

        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, checkpoint_list, epochs=5, batch_size=32):
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs,
            verbose=1,
            callbacks=checkpoint_list,
            validation_data=(self.x_val, self.y_val),
            batch_size=batch_size
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
    dataset_path = os.path.join(base, "wikiart")
    categories = os.listdir(dataset_path)
    create_image_db(100000000, dataset_path, categories, (224, 224), db_path)


def test_model_load():
    model = keras.models.load_model('model.h5')
    model.evalutate()


def dense_net_test():
    db_path = os.path.join(get_project_root(), 'images.hdf5')
    images = Images()
    categories = [
        'Color_Field_Painting',
        'Post_Impressionism',
        'Baroque',
        'Realism',
        'Rococo',
        'Fauvism',
        'Cubism',
        'Naive_Art_Primitivism',
        'Ukiyo_e',
        'Art_Nouveau_Modern',
        'Pop_Art'
    ]
    images.load(db_path, categories)

    dn = DenseNet()
    # Load images and create training and split data.
    dn.load_data(images.images, cat_len=20000)
    print(dn.amount_categories)
    # Creates Model
    dn.setup()
    # checkpoints
    anne = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-3)
    checkpoint = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
    checkpoint_list = [anne, checkpoint]
    # train class
    dn.train(checkpoint_list, epochs=10, batch_size=128)
    print(eval_model(dn))


def eval_model(model):
    y_pred = model.predict(model.x_test)
    total = 0
    accurate = 0
    accurate_index = []
    wrong_index = []

    for i, _ in enumerate(y_pred):
        prediction = np.argmax(y_pred[i])
        truth = np.argmax(model.y_test[i])
        if prediction == truth:
            accurate += 1
            accurate_index.append(i)
        else:
            wrong_index.append(i)
        total += 1

    return accurate / total


def evaluate_test():
    dn = DenseNet()
    dn.load('model.h5')
    with open('test_data.pickle', "rb") as p_file:
        test_data = pickle.load(p_file)

    dn.x_test = test_data[0]
    dn.y_test = test_data[1]
    print(eval_model(dn))

    y_pred = dn.predict(dn.x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(dn.y_test, axis=1)
    print(y_pred)
    print(y_true)
    
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    

if __name__ == '__main__':
    # gpu_test()
    # create_db()
    dense_net_test()
    # evaluate_test()
