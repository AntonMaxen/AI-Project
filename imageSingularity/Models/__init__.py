import abc


class IsModel(metaclass=abc.ABCMeta):
    def __init__(self):
        self.images = None

    @abc.abstractmethod
    def setup(self):
        pass

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def predict(self, test_values):
        pass


if __name__ == '__main__':
    pass