import os
from PIL import Image
import numpy as np


def load_images(amount, folder_name, categories, size):
    images = {}
    for category in categories:
        image_list = []
        path = os.path.join(folder_name, category)
        for i, filename in enumerate(os.listdir(path)):
            if i <= amount:
                fullpath = os.path.join(path, filename)
                img_data = Image.open(fullpath)
                img_data = img_data.resize(size)
                img_data = np.asarray(img_data)
                image_list.append(img_data)
            else:
                break

        images[category] = image_list

    return images


if __name__ == '__main__':
    pass
