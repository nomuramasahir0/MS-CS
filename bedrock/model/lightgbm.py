from bedrock.model import model
import lightgbm as lgb
from bedrock.model import util


class LGBMBinaryClassifier(model.Model):

    def __init__(self, theta, seed):
        super().__init__(theta)
        self.default = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'n_estimators': 100,
            'verbose': -1,
            'random_state': seed
        }
        theta = util.complement_default(self.default, theta)
        self.gbm = lgb.LGBMClassifier(**theta)

    def fit(self, X, y):
        self.gbm.fit(X, y)

    def pred(self, X):
        return self.gbm.predict(X)

    def pred_proba(self, X):
        return self.gbm.predict_proba(X)
