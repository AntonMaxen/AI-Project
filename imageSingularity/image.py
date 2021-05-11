import numpy as np


class Image:
    def __init__(self, image):
        self.pixels = np.array(image)


class Images:
    def __init__(self):
        pass

    def load(self):
        pass


class Model:
    def __init__(self, images):
        self.images = images
        pass

    def train(self):
        pass

    def predict(self):
        pass


class DenseNet(Model):
    def __init__(self):
        super(DenseNet).__init__()


def main():
    my_image = Image([1, 2, 3, 4])
    my_image.elements = my_image.elements.reshape((2,2))
    print(my_image.elements)


    #images = Images()
    #model = Model(images)
    #model.train()
    #model.predict()


if __name__ == '__main__':
    main()
