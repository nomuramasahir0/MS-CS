from abc import ABCMeta, abstractmethod


class FunctionBase(metaclass=ABCMeta):
    def __init__(self, dim, name):
        self.dim = dim
        self.name = name
        self.space = None
        self.estimator_type = None

    def set_space(self, space):
        self.space = space

    def convert_to_dict_space_from_normalized_space(self, x):
        assert self.space is not None
        theta = dict()
        for i in range(self.dim):
            theta[self.space.continuous_space[i].label] = self.space.continuous_space[
                i
            ].convert(x[i])
        return theta

    @abstractmethod
    def eval_naive(self, theta):
        pass

    @abstractmethod
    def eval_upper(self, theta, flag_target_evaluate=False):
        pass

    @abstractmethod
    def eval_unbiased(self, theta):
        pass

    @abstractmethod
    def eval_vr(self, theta):
        pass

    def evaluate(self, x):
        theta = self.convert_to_dict_space_from_normalized_space(x)
        # evaluation
        if self.estimator_type == "naive":
            return self.eval_naive(theta)
        elif self.estimator_type == "upper":
            return self.eval_upper(theta)
        elif self.estimator_type == "unbiased":
            return self.eval_unbiased(theta)
        elif self.estimator_type == "vr":
            return self.eval_vr(theta)
        else:
            raise NotImplementedError

    def target_evaluate(self, theta):
        return self.eval_upper(theta, flag_target_evaluate=True)
