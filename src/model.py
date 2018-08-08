import matplotlib as mpl
mpl.use('Agg')

from .data import *
from .tf_util import *
from .util import delete_if_exists
from .optimize import Optimizer

import tensorflow as tf
from pyrocko import guts
from pyrocko.guts import Object, Float, Bool

import logging
import shutil
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('pinky.model')


class Model(Object):

    hyperparameter_optimizer = Optimizer.T(optional=True)
    data_generator = DataGeneratorBase.T()
    evaluation_data_generator = DataGeneratorBase.T(optional=True)
    dropout_rate = Float.T(optional=True)
    batch_size = Int.T(default=10)
    n_epochs = Int.T(default=1)
    outdir = String.T(default='/tmp/dnn-seis')
    auto_clear = Bool.T(default=True)
    summary_outdir= String.T(default='summary')
    summary_nth_step = Int.T(default=1)
    shuffle_size = Int.T(
        optional=True, help='if set, shuffle examples at given buffer size.')
    tf.logging.set_verbosity(tf.logging.INFO)

    def __init__(self, tf_config=None, **kwargs):
        super().__init__(**kwargs)

        if self.auto_clear:
            delete_if_exists(self.summary_outdir)
            delete_if_exists(self.outdir)

        self.training_hooks = None
        self.devices = ['/device:GPU:0', '/device:GPU:1']
        self.tf_config = tf_config
        self.debug = logger.getEffectiveLevel() == logging.DEBUG
        # self.debug = logging.getLevel() == logging.DEBUG
        self.sess = tf.Session(config=tf_config)

        self.initializer = tf.truncated_normal_initializer(
            mean=0.1, stddev=0.1)

    def generate_eval_dataset(self):
        dataset = self.evaluation_data_generator.get_dataset()
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def generate_dataset(self):
        dataset = self.data_generator.get_dataset()
        if self.shuffle_size is not None:
            dataset = dataset.shuffle(buffer_size=self.shuffle_size)
        dataset = dataset.repeat(count=self.n_epochs)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def time_axis_cnn(self, input, n_filters, kernel_height=None,
            kernel_width=1, name=None, training=False):
        '''
        CNN along horizontal axis

        :param n_filters: number of filters
        :param kernel_height: convolution kernel size accross channels
        :param kernel_width: convolution kernel size accross time axis

        (Needs some debugging and checking)
        '''
        _, n_channels, n_samples, _ = input.shape

        if kernel_height is None:
            kernel_height = n_channels

        # with tf.variable_scope('conv_layer%s' %name):
        input = tf.layers.conv2d(
            inputs=input,
            filters=n_filters,
            kernel_size=(kernel_height, kernel_width),
            activation=self.activation,
            bias_initializer=self.initializer,
            name=name+'conv2d',
            )

        input = tf.layers.batch_normalization(input, training=training)
        input = tf.layers.max_pooling2d(
            input,
            pool_size=(kernel_height, kernel_width),
            strides=(1, 1),
            name=name+'max_pooling2d',
        )

        if self.debug:
            # super expensive!!
            logging.warn('Debug mode enables super expensive summaries.')
            tf.summary.image(
                'post-%s' % name, tf.split(
                    input, num_or_size_splits=n_filters, axis=-1)[0])

        return input

    def model(self, features, labels, mode, params):

        training = bool(mode == tf.estimator.ModeKeys.TRAIN)

        n_filters = params.get('base_capacity', 16)
        n_filters_factor = params.get('n_filters_factor', 2)
        kernel_width = params.get('kernel_width', 2)
        kernel_height = params.get('kernel_height', 2)
        kernel_width_factor = params.get('kernel_width_factor', 1)
        self.activation = params.get('activation', tf.nn.relu)
        n_channels, n_samples = self.data_generator.tensor_shape
        input = tf.reshape(features, [-1, n_channels, n_samples,  1])
        if self.debug: 
            view = features[:3]
            view = tf.expand_dims(view, -1)
            tf.summary.image('input', view)

        for ilayer in range(params.get('n_layers', 2)):
            n_filters = n_filters * (ilayer+1) * n_filters_factor
            input = self.time_axis_cnn(input,
                    n_filters=int(n_filters),
                    kernel_height=kernel_height,
                    kernel_width=int(kernel_width + ilayer*kernel_width_factor),
                    name='conv_%s' % ilayer)

        fc = tf.contrib.layers.flatten(input)
        fc = tf.layers.dense(fc, params.get('n_filters_dense', 32),
                name='dense', activation=self.activation,)

        dropout = params.get('dropout_rate', self.dropout_rate)
        if dropout is not None:
            fc = tf.layers.dropout(
                fc, rate=params['dropout_rate'], training=training)

        predictions = tf.layers.dense(fc, self.data_generator.n_classes)
        variable_summaries(predictions, 'predictions')
        predictions = tf.transpose(predictions) 
        labels = tf.transpose(labels)
        errors = predictions - labels

        variable_summaries(tf.abs(errors[0]), 'error_abs_x')
        variable_summaries(tf.abs(errors[1]), 'error_abs_y')
        variable_summaries(tf.abs(errors[2]), 'error_abs_z')

        # loss_carthesian = tf.sqrt(tf.reduce_sum(errors ** 2, axis=1, keepdims=False))
        # variable_summaries(loss_carthesian, 'training_loss')
        # loss = tf.reduce_mean(loss_carthesian)
        loss = tf.losses.mean_squared_error(labels, predictions)
        tf.summary.scalar('RMSE', loss)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(
                    learning_rate=params.get('learning_rate', 1e-4))
            train_op = optimizer.minimize(
                    loss=loss, global_step=tf.train.get_global_step())

            logging_hook = tf.train.LoggingTensorHook(
                    {"loss": loss, "step": tf.train.get_global_step()},
                    every_n_iter=10)

            return tf.estimator.EstimatorSpec(
                    mode=mode, loss=loss,
                    train_op=train_op,
                    # TODO: make this work for multi GPU:
                    evaluation_hooks=[self.get_summary_hook('eval')],
                    training_hooks=[
                            self.get_summary_hook('train'),
                            logging_hook,
                        ])

        elif mode == tf.estimator.ModeKeys.EVAL:

            metrics = {
                    'rmse_eval': tf.metrics.root_mean_squared_error(
                labels=labels, predictions=predictions, name='rmse_eval'),
                    'mse_eval': tf.metrics.mean_squared_error(
                labels=labels, predictions=predictions, name='mse_eval'),
                    'mae_eval': tf.metrics.mean_absolute_error(
                labels=labels, predictions=predictions, name='mae_eval')}


            with tf.name_scope('eval'):
                for k, v in metrics.items():
                    tf.summary.scalar(k, v[1])

            return tf.estimator.EstimatorSpec(
                    mode=mode, loss=loss, eval_metric_ops=metrics,
                    evaluation_hooks=[self.get_summary_hook('eval')])

    def get_summary_hook(self, subdir=''):
        return tf.train.SummarySaverHook(
            self.summary_nth_step,
            output_dir=os.path.join(self.summary_outdir, subdir),
            scaffold=tf.train.Scaffold(
                summary_op=tf.summary.merge_all())
        )

    def train_and_evaluate(self, params=None):
        params = params or {}
        with self.sess as default:

            est = tf.estimator.Estimator(
                model_fn=self.model, model_dir=self.outdir, params=params)

            train_spec = tf.estimator.TrainSpec(
                    input_fn=self.generate_dataset)

            eval_spec = tf.estimator.EvalSpec(
                    input_fn=self.generate_eval_dataset,
                    steps=None)

            tf.estimator.train_and_evaluate(est, train_spec, eval_spec)

    def train(self, params=None):
        '''Used by the optimizer.

        Todo: refactor to optimizer'''
        params = params or {}
        with self.sess as default:

            est = tf.estimator.Estimator(
                model_fn=self.model, model_dir=self.outdir, params=params)

            # New feature to test:
            # evaluator = tf.estimator.InMemoryEvaluatorHook(
            #     estimator=est,
            #     input_fn=self.generate_dataset
            #     )
            # est.train(input_fn=self.generate_dataset, hooks=[evaluator])
            
            est.train(input_fn=self.generate_dataset)
            logging.info('====== start evaluation')
            return est.evaluate(input_fn=self.generate_eval_dataset)

    def train_multi_gpu(self, params=None):
        ''' Use multiple GPUs for training.  Buggy...
        '''
        params = params or {}
        self.training_hooks = []
        # saver_hook = tf.train.CheckpointSaverHook()
        # summary_hook = tf.train.SummarySaverHook()
        with tf.train.MonitoredSession(
                # session_creator=tf.train.ChiefSessionCreator(),
                # hooks=[summary_hook]) as sess:
                # hooks=[saver_hook, summary_hook]) as sess:
                ) as sess:
            distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=2)
            run_config = tf.estimator.RunConfig(train_distribute=distribution)
            est = tf.estimator.Estimator(
                model_fn=self.model, model_dir=self.outdir, params=params,
                config=run_config
            )
            est.train(input_fn=self.generate_dataset)
            logging.info('====== start evaluation')
            return est.evaluate(input_fn=self.generate_eval_dataset)

    def optimize(self):
        if self.hyperparameter_optimizer is None:
            sys.exit('No hyperparameter optimizer defined in config file')
        self.hyperparameter_optimizer.optimize(self)


def main():
    import argparse

    parser = argparse.ArgumentParser(
                description='')
    parser.add_argument('--config')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--optimize', action='store_true')
    parser.add_argument('--write-tfrecord-model', metavar='FILENAME',
        help='write data_generator out to FILENAME')
    parser.add_argument('--from-tfrecord', metavar='FILENAME',
        help='read tfrecord')
    parser.add_argument('--write')
    parser.add_argument('--new-config')
    parser.add_argument('--show-data', action='store_true')
    parser.add_argument(
        '--cpu', action='store_true', help='force CPU usage')
    parser.add_argument('--ngpu', help='number of GPUs to use')
    parser.add_argument(
        '--debug', action='store_true')

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        # logging.setLevel(logging.DEBUG)
        logging.debug('Debug level active')

    if args.config:
        model = guts.load(filename=args.config)

    tf_config = None
    if args.cpu:
        tf_config = tf.ConfigProto(
            device_count={'GPU': 0}
        )

    if args.show_data:
        from . import plot
        plot.show_data(model, shuffle=True)

    elif args.write_tfrecord_model:
        import uuid
        model_id = uuid.uuid4()
        fn_tfrecord = '%s_train.tfrecord' % str(model_id)
        tfrecord_data_generator = DataGeneratorBase(fn_tfrecord=fn_tfrecord)
        tfrecord_data_generator.tensor_shape = model.data_generator.tensor_shape
        model.data_generator.write(fn_tfrecord)
        model.data_generator = tfrecord_data_generator

        if model.evaluation_data_generator is not None:
            fn_tfrecord = '%s_eval.tfrecord' % str(model_id)
            eval_data_generator = DataGeneratorBase(fn_tfrecord=fn_tfrecord)
            eval_data_generator.tensor_shape = model.data_generator.tensor_shape
            model.evaluation_data_generator.write(fn_tfrecord)
            model.evaluation_data_generator = eval_data_generator

        model.dump(filename=args.write_tfrecord_model)
        logging.info('Wrote new model file: %s' % args.write_tfrecord_model)

    elif args.from_tfrecord:
        logging.info('Reading data from %s' % args.from_tfrecord)
        model.data_generator = TFRecordData(fn_tfrecord=args.from_tfrecord)

    elif args.new_config:
        fn_config = args.new_config
        if os.path.exists(fn_config):
            print('file exists: %s' % fn_config)
            sys.exit()

        optimizer = Optimizer.get_example()
        model = Model(
            tf_config=tf_config,
            data_generator=GFSwarmData.get_example(),
            hyperparameter_optimizer=optimizer)

        model.regularize()

    if args.train and args.optimize:
        print('Can only use either --train or --optimize')
        sys.exit()

    if args.train:
        if args.ngpu:
            logging.info('Using %s GPUs' % args.ngpu)
            model.train_multi_gpu()
        else:
            model.train_and_evaluate()
            # model.train()

    elif args.optimize:
        model.optimize()
