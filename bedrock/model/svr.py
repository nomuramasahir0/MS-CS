from bedrock.model import model
from sklearn.svm import SVR


class SVMRegression(model.Model):

    def __init__(self, theta):
        super().__init__(theta)
        self.svr = SVR(kernel='rbf', C=theta['C'], gamma=theta['gamma'])

    def fit(self, X, y):
        self.svr.fit(X, y)

    def pred(self, X):
        return self.svr.predict(X)
