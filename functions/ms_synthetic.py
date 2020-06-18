import numpy as np

from functions.base import FunctionBase
from densratio import densratio


def create_one_task(mu, a, b, sigma, N):
    X, Y = [], []
    for _ in range(N):
        x = mu + sigma * np.random.randn()
        y = (a * x + b) + sigma * np.random.randn()
        X.append(x)
        Y.append(y)
    X = np.array(X).reshape(N, -1)
    Y = np.array(Y).reshape(N, -1)

    data = dict()
    data.update({'X': X, 'Y': Y})
    return data # X \in (N, dim), y \in (N, 1)


class MSSynthetic1D(FunctionBase):

    def __init__(self, args):
        super().__init__(1, 'MSSynthetic1D')
        np.random.seed(args.seed)

        self.num_sources = args.num_sources
        self.num_data = args.num_data
        source_mus = (np.random.rand(self.num_sources) * (args.source_mu_bound * 2.)) - args.source_mu_bound
        target_mu = (np.random.rand() * args.target_mu_bound * 2.) - args.target_mu_bound
        self.mus = np.append(source_mus, target_mu)
        assert len(self.mus) == self.num_sources + 1
        self.sigma = args.sigma

        self.estimator_type = args.estimator_type
        self.source_ind = np.random.randint(0, self.num_sources) # used for naive

        self.sources = [create_one_task(self.mus[s], args.coef_a, args.coef_b, self.sigma, args.num_data)
                        for s in range(self.num_sources)]
        self.target = create_one_task(self.mus[self.num_sources], args.coef_a, args.coef_b, args.sigma, args.num_data)

        print(f"target_mu:{target_mu}, source_mus:{source_mus}")
        # for unbiased or vr
        self.density_ratios = None
        if self.estimator_type == 'unbiased' or self.estimator_type == 'vr':
            hp_search_range = [0.1] if args.debug else [0.001, 0.01, 0.1, 1.0]
            self.density_ratios =\
                [densratio(self.target['X'], self.sources[s]['X'], alpha=0.,
                           sigma_range=hp_search_range, lambda_range=hp_search_range)
                 for s in range(self.num_sources)]

        # check consistency for mu and lambda
        if self.estimator_type == 'vr':
            self.log = dict()
            self.log['target_mu'] = target_mu
            self.log['source1_mu'] = self.mus[0]
            self.log['source2_mu'] = self.mus[1]
            self.log['abs_source1_mu'] = abs(target_mu - self.mus[0])
            self.log['abs_source2_mu'] = abs(target_mu - self.mus[1])
            self.log['lambda1'] = []
            self.log['lambda2'] = []
        else:
            self.log = None

    @staticmethod
    def L(theta, y):
        return (theta - y)**2 / 2.

    def eval_naive(self, theta):
        loss = 0.
        for s in range(self.num_sources):
            for y in self.sources[s]['Y']:
                loss += self.L(theta['theta'], y[0])
        num_all_data = sum([len(self.sources[s]['X']) for s in range(self.num_sources)])
        return loss / num_all_data

    def eval_upper(self, theta, flag_target_evaluate=False):
        loss = 0.
        for y in self.target['Y']:
            loss += self.L(theta['theta'], y[0])
        return loss / len(self.target['Y'])

    def eval_unbiased(self, theta):
        loss = 0.
        num_all_data = sum([len(self.sources[s]['X']) for s in range(self.num_sources)])
        for s in range(self.num_sources):
            for i in range(len(self.sources[s]['X'])):
                weight = self.density_ratios[s].compute_density_ratio(self.sources[s]['X'][i])[0]
                loss += weight * self.L(theta['theta'], self.sources[s]['Y'][i][0])
        return loss / num_all_data

    def eval_vr(self, theta):
        # this includes some implementation techniques for acceleration,
        # but the final calculation result matches the original formula in the paper.
        wL = [0.] * self.num_sources
        wL_square = [0.] * self.num_sources
        for s in range(self.num_sources):
            for i in range(len(self.sources[s]['X'])):
                weight = self.density_ratios[s].compute_density_ratio(self.sources[s]['X'][i])[0]
                loss = self.L(theta['theta'], self.sources[s]['Y'][i][0])
                wL[s] += weight * loss
                wL_square[s] += (weight * loss) ** 2
        task_divs = [(wL_square[s] / len(self.sources[s]['X'])) - (wL[s] / len(self.sources[s]['X']))**2
                     for s in range(self.num_sources)]
        norm_lamb = sum([len(self.sources[s]['X']) / task_divs[s] for s in range(self.num_sources)])
        lambs = [1. / (task_divs[s] * norm_lamb) for s in range(self.num_sources)]
        # log
        self.log['lambda1'].append(lambs[0])
        self.log['lambda2'].append(lambs[1])
        return sum([lambs[s] * wL[s] for s in range(self.num_sources)])
