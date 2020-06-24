from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):
    def __init__(self, theta):
        self.theta = theta

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def pred(self, X):
        pass
