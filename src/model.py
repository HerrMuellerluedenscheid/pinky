
from .data import *
from .tf_util import *
from .util import delete_if_exists
from .optimize import Optimizer

import tensorflow as tf
from pyrocko import guts
from pyrocko.guts import Object, Float, Bool

import logging
import shutil

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

    def __init__(self, tf_config=None, **kwargs):
        super().__init__(**kwargs)

        if self.auto_clear:
            delete_if_exists(self.summary_outdir)
            delete_if_exists(self.outdir)

        self.tf_config = tf_config
        self.debug = logger.getEffectiveLevel() == logging.DEBUG
        self.sess = tf.Session(config=tf_config)

        # initializer = tf.truncated_normal_initializer(
        self.initializer = tf.random_normal_initializer(
            # mean=0.5, stddev=0.1)
            mean=0.0, stddev=0.1)

    def generate_input(self):
        ''' Generates data and labels. '''
        dataset = self.data_generator.get_dataset()
        dataset = dataset.batch(self.batch_size)
        if self.shuffle_size:
            dataset = dataset.shuffle(buffer_size=self.shuffle_size)
        dataset = dataset.repeat(count=self.n_epochs)
        dataset = dataset.prefetch(buffer_size=self.batch_size)

        return dataset.make_one_shot_iterator().get_next()

    def generate_evaluation_input(self):
        ''' Generatas evaluation data and labels.'''
        self.evaluation_data_generator.get_dataset()
        dataset = dataset.batch(self.batch_size)
        return dataset.make_one_shot_iterator().get_next()

    def time_axis_cnn(self, input, n_filters, cross_channel_kernel=None,
            kernel_width=1, name=None, training=False):
        '''
        CNN along horizontal axis

        :param n_filters: number of filters
        :param cross_channel_kernel: convolution kernel size accross channels
        :param kernel_width: convolution kernel size accross time axis

        (Needs some debugging and checking)
        TODO:
         - cross_channel_kernel
        '''
        _, n_channels, n_samples, _ = input.shape

        if cross_channel_kernel is None:
            cross_channel_kernel = n_channels

        with tf.variable_scope('conv_layer%s' %name):
            input = tf.layers.conv2d(
                inputs=input,
                filters=n_filters,
                kernel_size=(cross_channel_kernel, kernel_width),    # use identity (1) along channels
                activation=self.activation,
                bias_initializer=self.initializer,
                name=name+'conv2d')

            input = tf.layers.batch_normalization(input, training=training)
            input = tf.layers.max_pooling2d(
                input,
                pool_size=(cross_channel_kernel, kernel_width),       # (height, width)
                strides=(1, 2),
                name=name+'max_pooling2d',
            )

            if self.debug:
                # super expensive!!
                logging.warn('Debug mode enables super expensive summaries.')
                tf.summary.image(
                    'post-%s' % name, tf.split(
                        input, num_or_size_splits=n_filters, axis=-1)[0])
                variable_summaries(input, name)
        return input

    def model(self, features, labels, mode, params):

        training = bool(mode == tf.estimator.ModeKeys.TRAIN)

        n_filters = params.get('base_capacity', 32)
        kernel_width = params.get('kernel_width', 2)
        kernel_height = params.get('kernel_height', 2)
        kernel_width_factor = params.get('kernel_width_factor', 1)
        self.activation = params.get('activation', tf.nn.relu)
        n_channels, n_samples = self.data_generator.tensor_shape
        input = tf.reshape(features, [-1, n_channels, n_samples,  1])

        # if self.debug: 
        #     tf.summary.image('input', features)

        for ilayer in range(params.get('n_layers', 3)):
            input = self.time_axis_cnn(input, n_filters * (1 + ilayer),
                    cross_channel_kernel=kernel_height,
                    kernel_width=int(kernel_width + ilayer*kernel_width_factor),
                    name='conv_%s' % ilayer, training=training)

        fc = tf.contrib.layers.flatten(input)
        fc = tf.layers.dense(fc, params.get('n_filters_dense', 512),
                activation=self.activation)

        if params.get('dropout_rate', False):
            fc = tf.layers.dropout(
                fc, rate=params['dropout_rate'], training=training)
        elif self.dropout_rate:
            fc = tf.layers.dropout(
                fc, rate=self.dropout_rate, training=training)

        predictions = tf.layers.dense(fc, self.data_generator.n_classes)
        variable_summaries(predictions, 'predictions')

        errors = predictions - labels
        tf.summary.scalar('error_z_mean', tf.reduce_mean(errors[-1]))
        tf.summary.scalar('error_lateral_mean',
               tf.reduce_mean(tf.sqrt(tf.reduce_sum(errors[0: 1]**2,
                   keepdims=False))))
        loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum((predictions - labels) ** 2, axis=1,
            keepdims=False)))
        loss = tf.Print(loss, [loss], "Mean Euclidian Error [m]: ")
        # lossalternative= tf.losses.mean_squared_error(
        #     labels,
        #     predictions,
        #     weights=1.0,
        #     scope=None,
        #     loss_collection=tf.GraphKeys.LOSSES,
        #     reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
        # )
        # tf.summary.scalar('lossalternative', lossalternative)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(
                    learning_rate=params.get('learning_rate', 1e-4))
            train_op = optimizer.minimize(
                    loss=loss, global_step=tf.train.get_global_step())
            tf.summary.scalar('loss', loss)

            return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    train_op=train_op,
                    training_hooks=[self.get_summary_hook()])

        elif mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss)

    def get_summary_hook(self):
        return tf.train.SummarySaverHook(
            self.summary_nth_step,
            output_dir=self.summary_outdir,
            summary_op=tf.summary.merge_all()
        )

    def train(self, params=None):
        params = params or {}

        with self.sess as default:
            est = tf.estimator.Estimator(
                model_fn=self.model, model_dir=self.outdir, params=params)

            est.train(input_fn=self.generate_input)

    def evaluate(self, params=None):
        logging.info('====== start evaluation')
        params = params or {}
        if self.evaluation_data_generator is None:
            logging.warn(
                'No evaluation data generator set! Can\'t evalauate after training')
            return

        with self.sess as default:
            est = tf.estimator.Estimator(
                model_fn=self.model, model_dir=self.outdir, params=params)
            return est.evaluate(input_fn=self.generate_evaluation_input, steps=1,
                    params=params)

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
    parser.add_argument(
        '--debug', action='store_true')

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug('Debug level active')

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
        fn_tfrecord = 'dump_%s.tfrecord' % str(uuid.uuid4())

        tfrecord_data_generator = DataGeneratorBase(fn_tfrecord=fn_tfrecord)
        tfrecord_data_generator.tensor_shape = model.data_generator.tensor_shape

        model.data_generator.write(fn_tfrecord)
        model.data_generator = tfrecord_data_generator
        model.dump(filename=args.write_tfrecord_model)
        logger.info('Wrote new model file: %s' % args.write_tfrecord_model)

    elif args.from_tfrecord:
        logger.info('Reading data from %s' % args.from_tfrecord)
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
        model.train()
        model.evaluate()

    elif args.optimize:
        model.optimize()
