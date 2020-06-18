import numpy as np
import os
from os.path import dirname as up
from functions.base import FunctionBase
from densratio import densratio
from bedrock.model.util import get_model

from sklearn.metrics import log_loss


def is_our_estimator(estimator_type):
    return estimator_type == 'unbiased' or estimator_type == 'vr'


def create_XYdict(data, name):
    X = data[:, :-1]
    Y = np.expand_dims(data[:, -1], 1)
    assert len(X[0,:]) == 7, f"shape is {data.shape}"
    return {'filename': name, 'X': X, 'Y': Y}


def load_dataset(estimator_type,
                 data_dir,
                 target_name,
                 ratio_validation,
                 ratio_dre,
                 is_separate_source_dens
                 ):
    dir_path = os.path.join(up(up(os.path.abspath(__file__))), "dataset/" + data_dir)
    source_density_list = []
    source_train_list = []
    source_val_list = []
    target_opt = None
    target_val = None
    filename_list = sorted(os.listdir(dir_path))
    for i, filename in enumerate(filename_list):
        file_path = os.path.join(dir_path, filename)
        data = np.load(file_path)
        np.random.shuffle(data)
        validation_size = int(len(data) * ratio_validation)
        remaining_size = len(data) - validation_size

        if filename == target_name:
            target_opt = create_XYdict(data[:remaining_size, :], filename + '_opt')
            target_val = create_XYdict(data[remaining_size:, :], filename + '_val')
        else:
            if is_our_estimator(estimator_type):

                if is_separate_source_dens:
                    density_size = int(remaining_size * ratio_dre)
                    train_size = remaining_size - density_size
                    source_dens = data[:density_size, :]
                    source_train = data[density_size:density_size+train_size, :]
                    source_val = data[density_size+train_size:, :]
                    assert density_size+train_size == remaining_size, 'density_size+train_size is not remaining_size.'
                    # append to list
                    source_density_list.append(create_XYdict(source_dens, filename))
                    source_train_list.append(create_XYdict(source_train, filename))
                    source_val_list.append(create_XYdict(source_val, filename))
                else:
                    source_train = data[:remaining_size, :]
                    source_dens = data[:remaining_size, :]
                    source_val = data[remaining_size:, :]
                    # append to list
                    source_density_list.append(create_XYdict(source_dens, filename))
                    source_train_list.append(create_XYdict(source_train, filename))
                    source_val_list.append(create_XYdict(source_val, filename))
            else:
                source_train = data[:remaining_size, :]
                source_val = data[remaining_size:, :]
                # append to list
                source_train_list.append(create_XYdict(source_train, filename))
                source_val_list.append(create_XYdict(source_val, filename))

    assert target_opt is not None and target_val is not None, 'Target is None'
    if not is_our_estimator(estimator_type):
        assert len(source_density_list) == 0, 'D^density is not empty though not using our estimator.'

    return source_density_list, source_train_list, source_val_list, target_opt, target_val


class GvHD(FunctionBase):

    def __init__(self, args):
        super().__init__(dim=args.dim, name='Parkinson')

        # check requirements
        assert args.target_name is not None, 'target_name is None.'
        if is_our_estimator(args.estimator_type):
            assert args.ratio_dre is not None, 'DRE ratio is None.'
            assert args.is_separate_source_dens is not None, 'is_separate_source_dens is None.'

        self.seed = args.seed
        np.random.seed(self.seed)
        self.space = None

        self.estimator_type = args.estimator_type
        self.sources_density, self.sources_train, self.sources_val, self.target_opt, self.target_val =\
            load_dataset(self.estimator_type, args.data_dir, args.target_name,
                         args.ratio_validation, args.ratio_dre, args.is_separate_source_dens)
        self.source_num = len(self.sources_train)

        if self.estimator_type == 'naive':
            self.source_naive_ind = np.random.randint(0, len(self.sources_train))
            self.source_name_naive = self.sources_train[self.source_naive_ind]['filename']
            self.is_source_concat_for_naive = args.is_source_concat_for_naive
        else:
            self.source_naive_ind = None
            self.source_name_naive = None
            self.is_source_concat_for_naive = None

        self.density_ratios = None
        if is_our_estimator(args.estimator_type):
            if args.debug:
                self.density_ratios =\
                    [densratio(self.target_opt['X'], self.sources_density[s]['X'], alpha=args.alpha,
                               sigma_range=[1.0], lambda_range=[0.001])
                     for s in range(len(self.sources_density))]
            else:
                hp_search_range = [1e-3, 1e-2, 1e-1, 1e-0]
                self.density_ratios = \
                    [densratio(self.target_opt['X'], self.sources_density[s]['X'], alpha=args.alpha,
                               sigma_range=hp_search_range, lambda_range=hp_search_range)
                     for s in range(len(self.sources_density))]

        self.all_source_train_data = self.concat_all_sources_train()
        self.all_source_val_data = self.concat_all_sources_val()

        self.model = get_model(args.model)

    def concat_all_sources_train(self):
        X = np.concatenate([s['X'] for s in self.sources_train], axis=0)
        Y = np.concatenate([s['Y'] for s in self.sources_train], axis=0)
        return {'X': X, 'Y': Y}

    def concat_all_sources_val(self):
        X = np.concatenate([s['X'] for s in self.sources_val], axis=0)
        Y = np.concatenate([s['Y'] for s in self.sources_val], axis=0)
        return {'X': X, 'Y': Y}

    def set_space(self, space):
        self.space = space

    def eval_naive(self, theta):
        model = self.model(theta, self.seed)
        if self.is_source_concat_for_naive:
            model.fit(self.all_source_train_data['X'], self.all_source_train_data['Y'].flatten())
            y_pred = model.pred_proba(self.all_source_val_data['X'])
            fval = log_loss(self.all_source_val_data['Y'], y_pred)
        else:
            model.fit(self.sources_train[self.source_naive_ind]['X'], self.sources_train[self.source_naive_ind]['Y'].flatten())
            y_pred = model.pred_proba(self.sources_val[self.source_naive_ind]['X'])
            fval = log_loss(self.all_source_val_data['Y'], y_pred)
        return fval

    def eval_upper(self, theta, flag_target_evaluate=False):
        model = self.model(theta, self.seed)
        model.fit(self.target_opt['X'], self.target_opt['Y'].flatten())
        y_pred = model.pred_proba(self.target_val['X'])
        fval = log_loss(self.target_val['Y'], y_pred)
        return fval

    def eval_unbiased(self, theta):
        assert self.density_ratios is not None, 'Density Ratio does not exist.'
        model = self.model(theta, self.seed)
        model.fit(self.all_source_train_data['X'], self.all_source_train_data['Y'].flatten())
        loss = 0.
        num_all_val_data = sum([len(self.sources_val[s]['X']) for s in range(self.source_num)])
        weights_sum = 0.
        for s in range(self.source_num):
            preds = model.pred_proba(self.sources_val[s]['X'])
            weights = self.density_ratios[s].compute_density_ratio(self.sources_val[s]['X'])
            weights_sum += sum(weights)
            for i in range(len(self.sources_val[s]['X'])):
                loss += weights[i] * log_loss(np.array([self.sources_val[s]['Y'][i][0]]), np.array([preds[i]]), labels=[0, 1])
        fval = loss / num_all_val_data
        return fval

    def eval_vr(self, theta):
        assert self.density_ratios is not None, 'Density Ratio does not exist.'
        model = self.model(theta, self.seed)
        model.fit(self.all_source_train_data['X'], self.all_source_train_data['Y'].flatten())

        wL = [0.] * self.source_num
        wL_square = [0.] * self.source_num
        for s in range(self.source_num):
            preds = model.pred_proba(self.sources_val[s]['X'])
            weights = self.density_ratios[s].compute_density_ratio(self.sources_val[s]['X'])
            for i in range(len(self.sources_val[s]['X'])):
                loss = log_loss(np.array([self.sources_val[s]['Y'][i]]), np.array([preds[i]]), labels=[0, 1])
                wL[s] += weights[i] * loss
                wL_square[s] += (weights[i] * loss) ** 2
        task_divs = [(wL_square[s] / len(self.sources_val[s]['X'])) - (wL[s] / len(self.sources_val[s]['X']))**2
                     for s in range(self.source_num)]
        norm_lamb = sum([len(self.sources_val[s]['X']) / task_divs[s] for s in range(self.source_num)])
        lambs = [1. / (task_divs[s] * norm_lamb) for s in range(self.source_num)]
        assert np.all(np.array(lambs) >= 0), f'Minus value exists in lambda. lambda:{lambs}, task_divs:{task_divs}'
        fval = sum([lambs[s] * wL[s] for s in range(self.source_num)])
        return fval
