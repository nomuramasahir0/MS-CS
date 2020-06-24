import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod


class RunnerBase(metaclass=ABCMeta):
    def __init__(self, **params):
        self.params = params
        self.log = {}
        self.best_params = None
        self.best_fval = 1e10

    @abstractmethod
    def run(self):
        pass

    def save(self):
        if 'path' not in self.params:
            return

        if hasattr(self.params['obj_func'], 'get_params_with_key'):
            self.best_params = self.params['obj_func'].get_params_with_key(
                self.best_params)
        if isinstance(self.best_params, np.ndarray):
            self.best_params = self.best_params.tolist()

        df = pd.DataFrame(self.log)
        df.index.name = '#index'
        df.to_csv('%s/log.csv' % self.params['path'], sep=',')
