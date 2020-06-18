
class Model:
    def __init__(self, theta):
        self.theta = theta

    def fit(self, X, y):
        raise NotImplementedError

    def pred(self, X):
        raise NotImplementedError
