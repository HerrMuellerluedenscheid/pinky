# import matplotlib as mpl
# mpl.use('PDF')

import os
import tensorflow as tf
from .util import delete_if_exists
from skopt import gp_minimize
from skopt import dump as dump_result
from skopt import load as load_result
from skopt.space import Real, Categorical, Integer
# from skopt.plots import plot_convergence, plot_objective_2D
from skopt.plots import plot_convergence, plot_objective
from skopt.plots import plot_objective, plot_evaluations
import matplotlib.pyplot as plt
import logging
from pyrocko.guts import Object, Int, Float, List, Tuple, String

try:
    from skopt.plots import plot_histogram
    _plot_histogram_error = False
except ImportError as e:
    _plot_histogram_error = e
    logging.debug(e)

logger = logging.getLogger()


def to_skopt_real(x, name, prior):
    return Real(low=x[0], high=x[1], prior=prior, name=name)


class Optimizer(Object):

    learning_rate = Tuple.T(3, Float.T(), default=(1e-3, 1e-5, 1e-4))  # low, high, default
    n_calls = Int.T(default=50, help='number of test sets')
    path_out = String.T(default='optimizer-results', help='base path where to store results, plots and logs')

    def __init__(self, **kwargs):

        '''
        TODO:
         - optimize kernel heigth (cross_channel_kernel)

        '''
        super().__init__(**kwargs)
        self.model = None
        self.result = None
        self.fn_result = self.extend_path('result.optmz')
        # self.dimensions = [
        #     to_skopt_real(self.learning_rate, 'learning_rate', 'log-uniform')]
        self.optimizer_defaults = [
            ('learning_rate', 1e-4),
            ('base_capacity', 32),
            ('kernel_width', 2),
            ('kernel_width_factor', 2),
            ('n_filters_dense', 512),
            ('dropout_rate', 0.1),
        ]

        self.dimensions = [
                Real(low=1e-6, high=1e-2, prior='log-uniform',
                    name='learning_rate'),
                Integer(low=12, high=64, name='base_capacity'),
                Integer(low=1, high=5, name='kernel_width'),
                Real(low=1, high=3, prior='uniform', name='kernel_width_factor'),
                Integer(low=64, high=1024, name='n_filters_dense'),
                Real(low=0., high=0.4, prior='uniform', name='dropout_rate'),
                ]

        print(self.dimensions)

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
        logger.info('+' * 20)
        logger.info('evaluating next set of parameters:')
        base ='   {}: {}\n'
        for kv in params.items():
            logger.info(base.format(*kv))

    def evaluate(self, args):
        ''' wrapper to parse gp_minimize args to model.train'''
        args = dict(zip(self.optimizer_keys, args))
        self.model.outdir = self.log_dir_name(args)
        self.announce_test(args)
        try:
            return self.model.train(args)['loss']
        except tf.errors.ResourceExhaustedError as e:
            logger.warn(e)
            logger.warn('Skipping this test')

    def optimize(self, model):
        '''Calling this method to optimize a :py:class:`pinky.model.Model`
        instance. '''

        self.model = model
        if self.model.auto_clear:
            delete_if_exists(self.path_out)

        self.result = gp_minimize(
                func=self.evaluate,
                dimensions=self.dimensions,
                acq_func='EI',  # Expected Improvement
                n_calls=self.n_calls,
                x0=self.optimizer_values)

        # dump_result(self.result, self.fn_result)
        self.evaluate_result()
        self.plot_results()

    def ensure_result(self):
        ''' Load and set minimizer result.'''
        if self.result is None:
            if self.fn_result is None:
                logger.warn(
                    'Cannot load results from filename: %s' % self.fn_result)
            self.result = load_result(self.fn_result)

    def extend_path(self, *path):
        '''Prepend `self.path_out` to `path`.'''
        return os.path.join(self.path_out, *path)

    def evaluate_result(self):
        self.ensure_result()

        # best = self.result.space.point_to_dict(self.result.x)
        best = self.result.x
        logger.info('Best parameter set:')
        logger.info(best)

        logger.info('Best parameter loss:')
        logger.info(self.result.fun)

    def ensure_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def plot_results(self):
        '''Produce and save result plots. '''
        self.ensure_result()
        self.ensure_directory(self.extend_path('plots'))

        if _plot_histogram_error:
            logger.warn(_plot_histogram_error)
        else:
            for dim_name in self.optimizer_keys:
                fig, ax = plot_histogram(result=self.result) #, dimension_name=dim_name)
                fig.savefig(self.extend_path('plots/histogram_%s.pdf' % dim_name))

        ax = plot_objective(
            result=self.result,)
            # dimension_names=self.non_categorical_dimensions)
        fig = plt.gcf()
        fig.savefig(self.extend_path('plots/objectives.pdf'))

        ax = plot_evaluations(
            result=self.result,)
            # dimension_names=self.non_categorical_dimensions)
        fig = plt.gcf()
        fig.savefig(self.extend_path('plots/evaluations.pdf'))

    def log_dir_name(self, params):
        '''Helper function to transform `params` into a logging directory
        name.'''

        placeholders = '{}_{}_' * len(params)
        identifiers = []
        for k, v in params.items():
            identifiers.append(k[0:3])
            identifiers.append(v)

        placeholders = placeholders.format(*identifiers)

        log_dir = self.extend_path('tf_logs/' + placeholders)
        logger.info('Created new logging directory: %s' % log_dir)
        return log_dir

    @classmethod
    def get_example(cls):
        '''Get an example instance of this class.'''
        return cls()


if __name__ == '__main__':
    print(Optimizer.get_example())
