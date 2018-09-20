import os
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import copy
from collections import OrderedDict
import signal

from skopt import gp_minimize
from skopt import dump as dump_result
from skopt import load as load_result
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence, plot_objective
from skopt.plots import plot_objective, plot_evaluations
from pyrocko.guts import Object, Int, Float, List, Tuple, String

from .util import delete_if_exists, ensure_dir
from .data import name_to_class


try:
    from skopt.plots import plot_histogram
    _plot_histogram_error = False
except ImportError as e:
    _plot_histogram_error = e
    logging.debug(e)


def to_skopt_real(x, name, prior):
    return Real(low=x[0], high=x[1], prior=prior, name=name)


class Param(Object):
    name = String.T()
    _type = None
    target_attribute = String.T(default='model',
        help='Which of models parameter should be modified (e.g. `config`)')

    def make_parameter(self):
        return self._type(low=self.low, high=self.high, name=self.name)


class PCategorical(Param):
    prior = String.T(optional=True)
    categories = List.T(String.T())
    default = String.T()
    _type = Categorical
    def make_parameter(self):
        return self._type(name=self.name, prior=self.prior,
                categories=self.categories)


class PInteger(Param):
    low = Float.T()
    high = Float.T()
    default = Float.T()
    _type = Integer
    def make_parameter(self):
        return self._type(low=self.low, high=self.high, name=self.name)


class PReal(Param):
    low = Float.T()
    high = Float.T()
    default = Float.T()
    prior = String.T(default='uniform')
    _type = Real
    def make_parameter(self):
        return self._type(low=self.low, high=self.high, name=self.name,
                prior=self.prior)


class Optimizer(Object):

    n_calls = Int.T(default=50, help='number of test sets')
    path_out = String.T(default='optimizer-results',
            help='base path where to store results, plots and logs')

    params = List.T(Param.T(), default=[PReal(name='learning_rate', low=1e-6,
        high=1e-2, default=1e-4)])

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.model = None
        self.result = None
        self.best_model_dir = self.extend_path('best-model')
        self.fn_result = self.extend_path('result.optmz')
        self.best_loss = 9e99
        self.param_keys = [p.name for p in self.params]
        self.params_dict = OrderedDict()
        for p in self.params:
            self.params_dict[p.name] = p.make_parameter()

        self._config_operations = [p.name for p in self.params if
            p.target_attribute=='config']

        self.optimizer_defaults = [(p.name, p.default) for p in self.params]

        self._ncalls = 0

        signal.signal(signal.SIGINT, self.plot_results)

    def clear(self):
        '''delete former runs.'''
        shutil.rmtree(self.path_out)

    @property
    def dimensions(self):
        return [v for _, v in self.params_dict.items()]

    @property
    def optimizer_keys(self):
        return [k for (k, default) in self.optimizer_defaults]

    @property
    def optimizer_values(self):
        return [default for (k, default) in self.optimizer_defaults]

    @property
    def non_categorical_dimensions(self):
        '''Returns a list of non-categorical dimension names.'''
        return [dim.name for dim in self.dimensions if not \
                           isinstance(dim, Categorical)]

    def announce_test(self, params):
        '''Log a parameter test set. '''
        logging.info('+' * 20)
        logging.info('evaluating next set of parameters:')
        base ='   {}: {}\n'
        for kv in params.items():
            logging.info(base.format(*kv))

    def update_model(self, model, kwargs):
        '''Set config and model attributes by kwargs.
        Rather sloppy...
        '''
        new_config = copy.deepcopy(model.config)
        self._ncalls += 1
        model.name = 'opt_%s-' % self._ncalls + self.base_name
        for key, arg in kwargs.items():

            # choose which object to modify (model or model.config)
            if 'config' in key:
                key = key.split('.')[-1]
                want_modifiy = new_config
            else:
                want_modifiy = model

            # If name is a ChunkOperation subclass, instatiate an object of
            # that class
            attribute = name_to_class.get(arg, False)
            if attribute:
                # chunk operation found
                attribute = attribute()
            else:
                attribute = arg

            if not getattr(want_modifiy, key):
                raise Exception('No such parameter: %s' % key)

            setattr(want_modifiy, key, attribute)
            
        model.config = new_config

    def save_model(self, model):
        '''copy the `model` to the `best_model` directory.'''
        shutil.rmtree(self.best_model_dir)
        shutil.copytree(model.outdir, self.best_model_dir)

    def evaluate(self, args):
        ''' wrapper to parse gp_minimize args to model.train'''
        kwargs = dict(zip(self.optimizer_keys, args))
        self.announce_test(kwargs)
        self.update_model(self.model, kwargs)
        try:
            loss = self.model.train_and_evaluate()[0]['loss']
            if loss < self.best_loss:
                print('found a better loss at %s' % loss)
                print('kwargs: ', kwargs)
                self.save_model(self.model)
                self.best_loss = loss
            else:
                self.model.clear_model()
            return loss

        except tf.errors.ResourceExhaustedError as e:
            logging.warn(e)
            logging.warn('Skipping this test, loss = 9e9')
            return 9e9

    def set_model(self, model):
        logging.info('prefixing model output path to %s' % self.path_out)
        model.prefix = self.path_out
        self.model = model
        self.base_name = self.model.name

    def optimize(self, model):
        '''Calling this method to optimize a :py:class:`pinky.model.Model`
        instance. '''
        self.set_model(model)
        ensure_dir(self.best_model_dir)

        self.result = gp_minimize(
                func=self.evaluate,
                dimensions=self.dimensions,
                acq_func='EI',
                n_calls=self.n_calls,
                x0=self.optimizer_values)

        self.evaluate_result()
        self.plot_results()

    def ensure_result(self):
        ''' Load and set minimizer result.'''
        if self.result is None:
            if self.fn_result is None:
                logging.warn(
                    'Cannot load results from filename: %s' % self.fn_result)
            self.result = load_result(self.fn_result)

    def extend_path(self, *path):
        '''Prepend `self.path_out` to `path`.'''
        return os.path.join(self.path_out, *path)

    def evaluate_result(self):
        self.ensure_result()

        # best = self.result.space.point_to_dict(self.result.x)
        best = self.result.x
        logging.info('Best parameter set:')
        logging.info(best)

        logging.info('Best parameter loss:')
        logging.info(self.result.fun)

    def plot_results(self, *args):
        '''Produce and save result plots. '''
        # self.ensure_result()
        ensure_dir(self.extend_path('plots'))

        if _plot_histogram_error:
            logging.warn(_plot_histogram_error)
        else:
            for dim_name in self.optimizer_keys:
                fig, ax = plot_histogram(result=self.result) #, dimension_name=dim_name)
                fig.savefig(self.extend_path('plots/histogram_%s.pdf' % dim_name))

        # ax = plot_objective(result=self.result,)
            # dimension_names=self.non_categorical_dimensions)
        # fig = plt.gcf()
        # fig.savefig(self.extend_path('plots/objectives.pdf'))

        ax = plot_evaluations(
            result=self.result,)
            # dimension_names=self.non_categorical_dimensions)
        fig = plt.gcf()
        fig.savefig(self.extend_path('plots/evaluations.pdf'))

    @classmethod
    def get_example(cls):
        '''Get an example instance of this class.'''
        return cls()


if __name__ == '__main__':
    print(Optimizer.get_example())
