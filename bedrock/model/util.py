import copy


def complement_default(default, theta):
    config = copy.deepcopy(theta)
    for key, val in default.items():
        if not (key in config):
            config[key] = val
    return config


def get_model(name):
    if name == "LGBMBinaryClassifier":
        from bedrock.model.lightgbm import LGBMBinaryClassifier

        return LGBMBinaryClassifier
    elif name == "SVR":
        from bedrock.model.svr import SVMRegression

        return SVMRegression
    else:
        raise NotImplementedError
