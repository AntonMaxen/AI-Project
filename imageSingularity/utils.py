import h5py
from PIL import Image
import os
import numpy as np
from pathlib import Path


def create_image_db(amount, dataset_folder, categories, size, db_path):
    if os.path.isfile(db_path):
        os.remove(db_path)

    f = h5py.File(db_path, "w")
    for category in categories:
        image_list = []
        file_names = []
        path = os.path.join(dataset_folder, category)
        for i, filename in enumerate(os.listdir(path)):
            if i <= amount:
                fullpath = os.path.join(path, filename)
                img_data = Image.open(fullpath)
                img_data = img_data.resize(size)
                img_data = np.asarray(img_data)
                file_names.append(filename)
                image_list.append(img_data)
            else:
                break

        file_names = [name.encode("ascii", "ignore") for name in file_names]
        category_folder = f.create_group(category)
        category_folder.create_dataset(f'images',
                                       np.shape(image_list),
                                       dtype=h5py.h5t.STD_U8BE,
                                       data=image_list)
        category_folder.create_dataset(f'meta',
                                       np.shape(file_names),
                                       dtype=h5py.string_dtype(),
                                       data=file_names)

    f.close()


def load_images(db_file, categories):
    f = h5py.File(db_file, "r")
    images = {}
    for category in categories:
        image_group = np.array(f[f'{category}/images'])
        label_group = np.array(f[f'{category}/meta'])

        images[category] = {
            "images": image_group,
            "labels": label_group
        }

    f.close()
    return images


def get_project_root():
    return Path(__file__).parent.parent


def test_create_db_and_load_images():
    base = get_project_root()
    db_path = os.path.join(base, "hej.hdf5")
    dataset_path = os.path.join(base, "wikiart")
    categories = os.listdir(dataset_path)

    create_image_db(100, dataset_path, categories, (100, 100), db_path)
    images = load_images(db_path, categories)
    print(images['Realism']['images'])


def main():
    test_create_db_and_load_images()


if __name__ == '__main__':
    main()
