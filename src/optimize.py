import os
import shutil
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
        self.best_model_dir = self.extend_path('best-model')
        self.fn_result = self.extend_path('result.optmz')
        self.best_loss = 9e99
        # self.dimensions = [
        #     to_skopt_real(self.learning_rate, 'learning_rate', 'log-uniform')]
        self.optimizer_defaults = [
            ('learning_rate', 1e-4),
            ('base_capacity', 16),
            ('kernel_width', 3),
            ('kernel_height', 3),
            # ('kernel_width_factor', 2),
            ('n_filters_dense', 64),
            # ('n_layers', 2),
            ('dropout_rate', 0.1),
        ]

        self.dimensions = [
                Real(low=1e-6, high=1e-2, prior='log-uniform',
                    name='learning_rate'),
                Integer(low=8, high=32, name='base_capacity'),
                Integer(low=2, high=5, name='kernel_width'),
                Integer(low=2, high=5, name='kernel_height'),
                # Real(low=1, high=3, prior='uniform', name='kernel_width_factor'),
                Integer(low=16, high=128, name='n_filters_dense'),
                # Integer(low=1, high=3, name='n_layers'),
                Real(low=0., high=0.4, prior='uniform', name='dropout_rate'),
                ]

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
        for key, arg in kwargs.items():
            for candidate in [new_config, model]:
                setattr(candidate, key, arg)

        model.config = new_config

    def save_model(self, model):
        '''copy the `model` to the `best_model` directory.'''
        shutil.rmtree(self.best_model_dir)
        shutil.copy(model.outdir, self.best_model_dir)

    def evaluate(self, args):
        ''' wrapper to parse gp_minimize args to model.train'''
        kwargs = dict(zip(self.optimizer_keys, args))
        self.announce_test(kwargs)
        self.update_model(self.model, kwargs)
        try:
            # self.model.train_multi_gpu(kwargs)
            loss = self.model.train_and_evaluate()[0]['loss']
            if loss < self.best_loss:
                print('found a better loss at %s' % loss)
                print('kwargs: ', kwargs)
                self.save_model(self.model)
                self.best_loss = loss
            else:
                shutil.rmtree(self.model.outdir)
            return loss

        except tf.errors.ResourceExhaustedError as e:
            logging.warn(e)
            logging.warn('Skipping this test, loss = 9e9')
            return 9e9

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

    def ensure_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def plot_results(self):
        '''Produce and save result plots. '''
        self.ensure_result()
        self.ensure_directory(self.extend_path('plots'))

        if _plot_histogram_error:
            logging.warn(_plot_histogram_error)
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
        logging.info('Created new logging directory: %s' % log_dir)
        return log_dir

    @classmethod
    def get_example(cls):
        '''Get an example instance of this class.'''
        return cls()


if __name__ == '__main__':
    print(Optimizer.get_example())
