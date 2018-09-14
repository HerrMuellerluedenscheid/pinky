import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from .data import *
from .tf_util import *
from .util import delete_if_exists
from . import plot
from .config import PinkyConfig
from .optimize import Optimizer

import tensorflow as tf
import tempfile
from pyrocko import guts
from pyrocko.guts import Object, Float, Bool

import logging
import shutil
import os

logger = logging.getLogger('pinky.model')


class DumpHook(tf.train.SessionRunHook):
    def __init__(self, labels, predictions):
        self.saver = tf.train.Saver(labels, predictions)
        self.session = session

    def begin(self):
        pass

  # def before_run(self, run_context):
  #   return tf.train.SessionRunArgs(self.loss)  

    def after_run(self, run_context, run_values):
        # loss_value = run_values.results
        print(self.labels)
        print(self.predictions)
        self.saver.save(self.session, 'test-model')


class Layer(Object):
    name = String.T()
    n_filters = Int.T()

    activation = String.T(
            default='leaky_relu', help='activation function of tf.nn') # test
    # leaky relu

    def get_activation(self):
        '''Return activation function from tensorflow.nn'''
        return getattr(tf.nn, self.activation)


class CNNLayer(Layer):
    kernel_width = Int.T()
    kernel_height = Int.T(optional=True)
    _initializer = tf.truncated_normal_initializer(
            mean=0.1, stddev=0.1)

    def chain(self, input, training=False):
        _, n_channels, n_samples, _ = input.shape

        logger.debug('input shape %s' % input.shape)
        pool_height = self.kernel_height if self.kernel_height else 1
        kernel_height = self.kernel_height or n_channels
        print('CHAINGIN IN ', n_channels)

        input = tf.layers.conv2d(
            inputs=input,
            filters=self.n_filters,
            kernel_size=(kernel_height, self.kernel_width),
            activation=self.get_activation(),
            kernel_initializer=self._initializer,  # Test
            bias_initializer=self._initializer,
            name=self.name)

        input = tf.layers.max_pooling2d(input,
            pool_size=(pool_height, self.kernel_width),
            strides=(1, 1), name=self.name+'maxpool')
        
        if logger.getEffectiveLevel() == logging.DEBUG:
            tf.summary.image(
                'post-%s' % self.name, tf.split(
                    input, num_or_size_splits=self.n_filters, axis=-1)[0])

        return input


class DenseLayer(Layer):

    def chain(self, input, training=False):
        fc = tf.contrib.layers.flatten(input)
        return tf.layers.dense(fc, self.n_filters, name=self.name,
                activation=self.get_activation())


class Model(Object):

    name = String.T(default='unnamed',
        help='Used to identify the model and runs in summy and checkpoint \
            directory')

    config = PinkyConfig.T()
    hyperparameter_optimizer = Optimizer.T(optional=True)
    dropout_rate = Float.T(default=0.01)
    batch_size = Int.T(default=10)
    n_epochs = Int.T(default=1)
    max_steps = Int.T(default=5000)
    outdir = String.T(default=tempfile.mkdtemp(prefix='pinky-'))
    summary_outdir= String.T(default='summary')
    summary_nth_step = Int.T(default=1)

    shuffle_size = Int.T(
        optional=True, help='if set, shuffle examples at given buffer size.')

    tf.logging.set_verbosity(tf.logging.INFO)

    layers = List.T(Layer.T(), help='Define the model')

    def __init__(self, tf_config=None, **kwargs):
        super().__init__(**kwargs)

        self.devices = ['/device:GPU:0', '/device:GPU:1']
        self.tf_config = tf_config
        self.debug = logger.getEffectiveLevel() == logging.DEBUG
        self.sess = tf.Session(config=tf_config)
        self.est = None

    def extend_path(self, p):
        return os.path.join(p, self.name)

    def get_summary_outdir(self):
        return self.extend_path(self.summary_outdir)

    def get_outdir(self):
        return self.extend_path(self.outdir)

    def clear(self):
        ''' Delete summary and model directories.'''
        delete_if_exists(self.get_summary_outdir())
        delete_if_exists(self.get_outdir())

    def generate_eval_dataset(self):
        return self.config.evaluation_data_generator.get_dataset().batch(
                self.batch_size)

    def generate_dataset(self):
        dataset = self.config.data_generator.get_dataset()
        if self.shuffle_size is not None:
            dataset = dataset.shuffle(buffer_size=self.shuffle_size)

        return dataset.repeat(count=self.n_epochs).batch(self.batch_size)

    def model(self, features, labels, mode, params):
        training = bool(mode == tf.estimator.ModeKeys.TRAIN)

        if self.debug: 
            view = features[:3]
            view = tf.expand_dims(view, -1)
            tf.summary.image('input', view)

        # It's impo
        input = tf.reshape(features, [-1, *self.config.data_generator.tensor_shape,  1])
        # input = tf.reshape(features, [self.batch_size,
        #     self.config.data_generator.tensor_shape[0], -1,  1])

        # CHEKC BATCH SIZE@!!!!!!
        # input = tf.reshape(features, [-1,
        #     *self.config.data_generator.tensor_shape,  self.batch_size])

        for layer in self.layers:
            logger.debug('chain in layer: %s' % layer)
            input = layer.chain(input=input, training=training)

        # fc = tf.layers.dropout(
        #     input, rate=self.dropout_rate, training=training)
        # fc = tf.layers.batch_normalization(fc, training=training)

        predictions = input
        tf.summary.tensor_summary('predictions', predictions)

        predictions = tf.transpose(predictions) 
        labels = tf.transpose(labels)

        errors = tf.abs(predictions - labels)

        variable_summaries(errors[0], 'error_abs_x')
        variable_summaries(errors[1], 'error_abs_y')
        variable_summaries(errors[2], 'error_abs_z')

        # loss_carthesian = tf.sqrt(tf.reduce_sum(errors ** 2, axis=1, keepdims=False))
        # variable_summaries(loss_carthesian, 'training_loss')
        # loss = tf.reduce_mean(loss_carthesian)

        loss = tf.losses.mean_squared_error(labels, predictions)
        # loss_ = loss.eval(session=self.sess)
        # num.savetxt(loss_, 'xxx')
        tf.summary.scalar('loss', loss)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(
                    learning_rate=params.get('learning_rate', 1e-4))
                    # learning_rate=params.get('learning_rate', 0.0009))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(
                    loss=loss, global_step=tf.train.get_global_step())

            logging_hook = tf.train.LoggingTensorHook(
                    {"loss": loss, "step": tf.train.get_global_step()},
                    every_n_iter=10)
            return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    train_op=train_op,
                    # TODO: make this work for multi GPU:
                    evaluation_hooks=[
                            self.get_summary_hook('eval'),],
                    training_hooks=[
                            self.get_summary_hook('train'),
                            logging_hook,])

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
            # vloss = tf.Variable(loss)
            # restore_vars = [vloss]
            return tf.estimator.EstimatorSpec(
                    mode=mode, loss=loss, eval_metric_ops=metrics,
                    evaluation_hooks=[self.get_summary_hook('eval'),
                        # dump_hook
                        ])
            # saver.save(self.sess, '/tmp/testtt')

        elif mode == tf.estimator.ModeKeys.PREDICT:
            ...

    def get_summary_hook(self, subdir=''):
        return tf.train.SummarySaverHook(
            self.summary_nth_step,
            output_dir=os.path.join(self.get_summary_outdir(), subdir),
            scaffold=tf.train.Scaffold(
                summary_op=tf.summary.merge_all())
        )

    def train_and_evaluate(self, params=None):
        params = params or {}
        with self.sess as default:

            self.est = tf.estimator.Estimator(
                model_fn=self.model, model_dir=self.get_outdir(), params=params)

            train_spec = tf.estimator.TrainSpec(
                    input_fn=self.generate_dataset,
                    max_steps=self.max_steps)

            eval_spec = tf.estimator.EvalSpec(
                    input_fn=self.generate_eval_dataset,
                    steps=None)

            return tf.estimator.train_and_evaluate(self.est, train_spec, eval_spec)

    def train(self, params=None):
        '''Used by the optimizer.

        Todo: refactor to optimizer'''
        params = params or {}
        with self.sess as default:

            self.est = tf.estimator.Estimator(
                model_fn=self.model, model_dir=self.get_outdir(), params=params)

            # New feature to test:
            # evaluator = tf.estimator.InMemoryEvaluatorHook(
            #     estimator=est,
            #     input_fn=self.generate_dataset
            #     )
            # est.train(input_fn=self.generate_dataset, hooks=[evaluator])
            
            self.est.train(input_fn=self.generate_dataset)
            logger.info('====== start evaluation')
            return self.est.evaluate(input_fn=self.generate_eval_dataset)

    def train_multi_gpu(self, params=None):
        ''' Use multiple GPUs for training.  Buggy...'''
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
            logger.info('====== start evaluation')
            return est.evaluate(input_fn=self.generate_eval_dataset)

    def optimize(self):
        if self.hyperparameter_optimizer is None:
            sys.exit('No hyperparameter optimizer defined in config file')
        self.hyperparameter_optimizer.optimize(self)

    def show_kernels(self, layer_name):
        '''Shows weight kernels of the given `layer_name`.'''
        weights = self.est.get_variable_value('%s/kernel' % layer_name)
        plot.show_kernels(weights[:, :, 0, :])

    def restore(self):
        # tf.reset_default_graph()
        saver = tf.train.Saver()
        self.train_and_evaluate()
        with self.sess as sess:
            # return tf.estimator.train_and_evaluate(self.est, train_spec, eval_spec)
            ckpt = tf.train.get_checkpoint_state(self.get_outdir())
            x = saver.restore(sess, ckpt.model_checkpoint_path)
            print(x)


def main():
    import argparse

    parser = argparse.ArgumentParser(
                description='')
    parser.add_argument('--config')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--optimize', action='store_true')
    parser.add_argument('--write-tfrecord', metavar='FILENAME',
        help='write data_generator out to FILENAME')
    parser.add_argument('--from-tfrecord', metavar='FILENAME',
        help='read tfrecord')
    parser.add_argument('--new-config')
    parser.add_argument('--clear', help='delete remaints of former runs',
            action='store_true')
    parser.add_argument('--show-data', action='store_true')
    parser.add_argument(
        '--cpu', action='store_true', help='force CPU usage')
    parser.add_argument('--ngpu', help='number of GPUs to use')
    parser.add_argument(
        '--debug', action='store_true')
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--force', action='store_true')

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logging.debug('Debug level active')
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logger.setLevel(logging.DEBUG)

    if args.config:
        model = guts.load(filename=args.config)
        model.config.setup()

    if args.clear:
        model.clear()

    tf_config = None
    if args.cpu:
        tf_config = tf.ConfigProto(
            device_count={'GPU': 0}
        )

    if args.show_data:
        from . import plot
        plot.show_data(model, shuffle=True)
        plt.show()

    elif args.write_tfrecord:
        model_id = args.write_tfrecord

        if os.path.isfile(args.write_tfrecord):
            if args.force:
                delete_candidate = guts.load(filename=args.write_tfrecord)
                for g in [delete_candidate.config.evaluation_data_generator,
                        delete_candidate.config.data_generator]:
                    g.cleanup()
                delete_if_exists(args.write_tfrecord)
            else:
                print('file %s exists. use --force to overwrite' % \
                        args.write_tfrecord)
                sys.exit(0)

        fn_tfrecord = '%s_train.tfrecord' % str(model_id)
        tfrecord_data_generator = DataGeneratorBase(
                fn_tfrecord=fn_tfrecord, config=model.config)
        model.config.data_generator.write(fn_tfrecord)
        model.config.data_generator = tfrecord_data_generator

        fn_tfrecord = '%s_eval.tfrecord' % str(model_id)
        tfrecord_data_generator = DataGeneratorBase(
                fn_tfrecord=fn_tfrecord, config=model.config)

        model.config.evaluation_data_generator.write(fn_tfrecord)
        model.config.evaluation_data_generator = tfrecord_data_generator

        model.name += '_tf'
        model.dump(filename=args.write_tfrecord + '.config')
        logger.info('Wrote new model file: %s' % (
            args.write_tfrecord + '.config'))

    elif args.from_tfrecord:
        logger.info('Reading data from %s' % args.from_tfrecord)
        model.config.data_generator = TFRecordData(
                fn_tfrecord=args.from_tfrecord)

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
            logger.info('Using %s GPUs' % args.ngpu)
            model.train_multi_gpu()
        else:
            model.train_and_evaluate()
            model.show_kernels('conv_1')
            # model.train()
    elif args.restore:
        model.restore()

    elif args.optimize:
        model.optimize()
