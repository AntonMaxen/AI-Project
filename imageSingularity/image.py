import numpy as np
import os
from imageSingularity.utils import load_images, get_project_root


class Image:
    def __init__(self, image):
        self.pixels = np.array(image)


class Images:
    def __init__(self):
        self.images = None

    def load(self, db_path, categories):
        self.images = load_images(db_path, categories)


def main():
    #my_image = Image([1, 2, 3, 4])
    #my_image.elements = my_image.elements.reshape((2, 2))
    images = Images()
    db_path = os.path.join(get_project_root(), 'hej.hdf5')
    images.load(db_path)

    print(images.images)


    #images = Images()
    #model = Model(images)
    #model.train()
    #model.predict()


if __name__ == '__main__':
    main()