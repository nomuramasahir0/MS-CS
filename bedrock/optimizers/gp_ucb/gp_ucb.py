import numpy as np
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize
from bedrock.optimizers.utils import opt_util


class GPUCB:

    def create_theta(self, x):
        theta = dict()
        for i in range(self.dim):
            theta[self.space.continuous_space[i].label] = self.space.continuous_space[i].convert(x[i])
        return theta

    def log_generation(self, x, y):
        self.evals += 1
        self.fval = y
        theta = self.create_theta(x)
        print("theta:{}, fval:{}".format(theta, self.fval))
        if self.fval < self.best_fval:
            self.best_fval = self.fval
            self.best_params = theta

        self.log['evals'].append(self.evals)
        self.log['fval'].append(self.fval)
        self.log['best_fval'].append(self.best_fval)
        for i in range(self.dim):
            self.log[self.space.continuous_space[i].label].append(self.space.continuous_space[i].convert(x[i]))

    def __init__(self, **params):
        self.seed = params['seed']
        np.random.seed(self.seed)
        self.B = params['B']
        self.obj_func = params['obj_func']
        self.dim = params['obj_func'].dim
        # GP
        self.xs = np.zeros((0, self.dim))
        self.ys = np.zeros((0, 1))
        # kernel
        matern = Matern(nu=2.5, length_scale=[1. for _ in range(self.dim)])
        white = WhiteKernel(noise_level=1.)
        constant = ConstantKernel(constant_value=1.)
        self.gp = GaussianProcessRegressor(
            kernel=constant * matern + white,
            normalize_y=True,
            n_restarts_optimizer=15
        )
        # BO
        self.beta = 2.
        self.num_initial_samples = 5
        self.acq_initial_points = 50

        # for logging
        self.evals = 0
        self.fval = None
        self.best_params = None
        self.best_fval = np.inf
        self.space = params['space']
        self.log = opt_util.basic_log_setup(self.space)

    def ucb(self, x):
        mean, std = self.gp.predict(x, return_std=True)
        # Note that our implementation targets minimization, so this is lcb, in fact.
        return mean - self.beta * std

    def argmin_acq(self):
        x_initials = [np.random.uniform(0., 1., size=self.dim) for _ in range(self.acq_initial_points)]
        x_best = None
        fval_best = np.inf
        for xp in x_initials:
            res = minimize(lambda x: self.ucb(x.reshape(1, -1)), # (1, -1) means it contains a single sample
                           xp.reshape(1, -1),
                           bounds=[(0., 1.) for _ in range(self.dim)],
                           method='L-BFGS-B')
            if res.fun < fval_best:
                x_best = res.x
                fval_best = res.fun
        return x_best

    def optimize(self):
        for t in range(self.B):
            # Step 1. select sample to evaluate
            if t < self.num_initial_samples:
                x = np.random.uniform(0., 1., size=self.dim)
            else:
                x = self.argmin_acq()

            # Step 2. evaluate the sample selected in Step 1.
            y = self.obj_func.evaluate(x)
            self.log_generation(x, y)

            # Step 3. accumulate the observed data
            self.xs = np.concatenate((self.xs, np.array(x).reshape(1, self.dim)))
            self.ys = np.concatenate((self.ys, np.array(y).reshape(1, 1)))
