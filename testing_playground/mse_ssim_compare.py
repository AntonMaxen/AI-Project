# USAGE
# python compare.py

# import the necessary packagesi
import os

from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import cv2
import numpy as np

FOLDERS = ['surrealism', 'impressionism']
ROOT = 'data'

ingroup_mse = []
ingroup_ssim = []
between_mse = []
between_ssim = []


def mse(a, b):
    test = a.shape
    print()


    return np.sum((a.astype("float") - b.astype("float")) ** 2) / float(a.shape[0] * a.shape[1])



def compare_images(a, b):
    image_a, name_a = a[0], a[1]
    image_b, name_b = b[0], b[1]

    mse_result = mse(image_a, image_b)
    #  använd multichannel=True för att jämföra bilder med färger
    ssim_result = ssim(image_a, image_b, multichannel=True)

    if name_a == name_b:
        ingroup_mse.append(mse_result)
        ingroup_ssim.append(ssim_result)
    else:
        between_mse.append(mse_result)
        between_ssim.append(ssim_result)

    fig = plt.figure()
    plt.suptitle(f"{name_a} vs {name_b}\n\nMSE: {round(mse_result, 2)}, SSIM: {round(ssim_result, 2)}")

    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(image_a, cmap=plt.cm.gray)
    plt.axis("off")

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(image_b, cmap=plt.cm.gray)
    plt.axis("off")
    plt.show()


def resize(images):
    images_fixed = []

    width = min([i.shape[0] for i in images])
    height = min([i.shape[1] for i in images])

    for image in images:
        w = int(round((width / image.shape[0]) * image.shape[0]))
        h = int(round((height / image.shape[1]) * image.shape[1]))
        images_fixed.append(cv2.resize(image, (w, h)))

    return images_fixed


def crop(images):
    images_fixed = []

    width = min([i.shape[0] for i in images])
    height = min([i.shape[1] for i in images])

    for image in images:
        images_fixed.append(image[:width, :height])

    return images_fixed


def load_images_from_folder():
    images = []
    names = []
    for folder in FOLDERS:
        for image in os.listdir(f'{ROOT}/{folder}'):
            images.append(cv2.imread(f'{ROOT}/{folder}/{image}'))
            names.append(folder.capitalize())
    return images, names


def mean(n, r=2):
    return round(sum(n) / len(n), r)


def main():
    images, names = load_images_from_folder()
    images = resize(images)
    # images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    # initialize the figure
    fig = plt.figure("Images")

    for (i, (name, image)) in enumerate(zip(names, images)):
        # show the image
        ax = fig.add_subplot(1, len(images), i + 1)
        ax.set_title(name)
        plt.imshow(image, cmap=plt.cm.gray)
        plt.axis("off")
    plt.show()

    import itertools
    for i, combos in enumerate(itertools.combinations(zip(images, names), 2)):
        compare_images(combos[0], combos[1])


    ingroup_mse_result = mean(ingroup_mse)
    ingroup_ssim_result = mean(ingroup_ssim)
    between_mse_result = mean(between_mse)
    between_ssim_result = mean(between_ssim)


    print(f"Within-group variance:\n")
    print(f"Mean squared error: {ingroup_mse_result}")
    print(f"Structural similarity index: {ingroup_ssim_result}\n")

    print(f"Between group variance:\n")
    print(f"Mean squared error: {between_mse_result}")
    print(f"Structural similarity index: {between_ssim_result}")



if __name__ == '__main__':
    main()
