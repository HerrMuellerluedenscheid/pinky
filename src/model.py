from .data import *
from .tf_util import *
from .util import delete_if_exists, ensure_dir
from .config import PinkyConfig
from .optimize import Optimizer
from . import plot

import functools
import tensorflow as tf
import tempfile
from pyrocko import guts, util as putil, io
from pyrocko import trace
from pyrocko.guts import Object, Float, Bool
from pyrocko.gui.marker import Marker, EventMarker
import pyrocko.model as pmodel

import logging
import shutil
import os

logger = logging.getLogger('pinky.model')


class Layer(Object):
    '''A 2D CNN followed by dropout and batch normalization'''

    name = String.T(help='Identifies the model')
    n_filters = Int.T(help='Number of output filters')
    activation = String.T(
            default='leaky_relu', help='activation function of tf.nn')

    def get_activation(self):
        '''Return activation function from tensorflow.nn'''
        return getattr(tf.nn, self.activation)

    def visualize_kernel(self, estimator, index=0, save_path=None):
        '''Subclass this method for plotting kernels.'''
        logger.debug('not implemented')

    def chain(self, input, levels, training=False, **kwargs):
        '''Subclass this method to attach the layer to a sequential model.'''
        pass


class CNNLayer(Layer):

    '''A 2D CNN followed by dropout and batch normalization'''
    kernel_width = Int.T()
    kernel_height = Int.T(optional=True)

    pool_width = Int.T()
    pool_height = Int.T()

    dilation_rate = Int.T(default=0)

    strides = Tuple.T(2, Int.T(), default=(1, 1))
    tf_fun = tf.layers.conv2d

    is_detector= Bool.T(default=False)

    def chain(self, input, level, training=False, dropout_rate=0.):
        _, n_channels, n_samples, _ = input.shape

        logger.debug('input shape %s' % input.shape)
        kernel_height = self.kernel_height or n_channels

        kwargs = {}
        if self.dilation_rate:
            kwargs.update({'dilation_rate': self.dilation_rate})

        inpower = tf.sqrt(tf.reduce_sum(tf.square(input)))

        input = tf.layers.conv2d(
            inputs=input,
            filters=self.n_filters,
            kernel_size=(kernel_height, self.kernel_width),
            activation=self.get_activation(),
            strides=self.strides,
            name=self.name,
            **kwargs)

        input = tf.layers.max_pooling2d(input,
            pool_size=(self.pool_width, self.pool_height),
            strides=(1, 1), name=self.name+'maxpool')

        if logger.getEffectiveLevel() == logging.DEBUG:
            tf.summary.image(
                'post-%s' % self.name, tf.split(
                    input, num_or_size_splits=self.n_filters, axis=-1)[0])

        if self.is_detector:
            batch_mean = tf.reduce_mean(input, reduction_indices=(1, 2, 3),
                    keepdims=True)

            stdev = tf.sqrt(tf.reduce_mean(tf.square(input - batch_mean),
                    reduction_indices=(1, 2, 3)))

            level += (tf.sqrt(tf.reduce_sum(tf.square(input))) / inpower)

        input = tf.layers.dropout(
            input, rate=dropout_rate, training=training)

        return tf.layers.batch_normalization(input, training=training), level

    def visualize_kernel(self, estimator, index=0, save_path=None, **kwargs):
        save_name = pjoin(save_path, 'kernel-%s' % self.name)
        weights = estimator.get_variable_value('%s/kernel' % self.name)
        plot.show_kernels(weights[:, :, index, ...], name=save_name)


class SeparableCNNLayer(CNNLayer):

    def chain(self, input, level, training=False, dropout_rate=0.):
        _, n_channels, n_samples, _ = input.shape

        logger.debug('input shape %s' % input.shape)
        kernel_height = self.kernel_height or n_channels

        input = tf.layers.separable_conv2d(
            inputs=input,
            filters=self.n_filters,
            kernel_size=(kernel_height, self.kernel_width),
            activation=self.get_activation(),
            strides=self.strides,
            name=self.name)

        input = tf.layers.max_pooling2d(input,
            pool_size=(self.pool_width, self.pool_height),
            strides=(1, 1), name=self.name+'maxpool')

        if logger.getEffectiveLevel() == logging.DEBUG:
            tf.summary.image(
                'post-%s' % self.name, tf.split(
                    input, num_or_size_splits=self.n_filters, axis=-1)[0])

        input = tf.layers.dropout(
            input, rate=dropout_rate, training=training)

        return tf.layers.batch_normalization(input, training=training), level


class DenseLayer(Layer):

    def chain(self, input, level, training=False, **kwargs):
        input = tf.contrib.layers.flatten(input)
        return tf.layers.dense(input, self.n_filters, name=self.name,
                activation=self.get_activation()), level

    def visualize_kernel(self, estimator, index=0, save_path=None):
        save_name = pjoin(save_path, 'kernel-%s' % self.name)
        weights = estimator.get_variable_value('%s/kernel' % self.name)
        plot.show_kernels_dense(weights, name=save_name)


class Model(Object):

    name = String.T(default='unnamed',
        help='Used to identify the model and runs in summy and checkpoint \
            directory')

    config = PinkyConfig.T()
    learning_rate = Float.T(default=1e-3)
    dropout_rate = Float.T(default=0.01)
    batch_size = Int.T(default=10)
    n_epochs = Int.T(default=1)
    max_steps = Int.T(default=5000)
    outdir = String.T(default=tempfile.mkdtemp(prefix='pinky-'))
    summary_outdir = String.T(default='summary')
    summary_nth_step = Int.T(default=1)

    shuffle_size = Int.T(
        optional=True, help='if set, shuffle examples at given buffer size.')

    tf.logging.set_verbosity(tf.logging.INFO)
    force_dropout = Bool.T(optional=True)
    layers = List.T(Layer.T(), help='A list of `Layers` instances.')

    def __init__(self, tf_config=None, **kwargs):
        ''' '''
        super().__init__(**kwargs)

        if not tf_config:
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=tf_config)
        self.debug = logger.getEffectiveLevel() == logging.DEBUG
        self.est = None
        self.prefix = None
        self.tinc_detect = 1.

    def set_tfconfig(self, tf_config):
        self.sess.close()
        self.sess = tf.Session(config=tf_config)

    def enable_debugger(self):
        '''wrap session to enable breaking into debugger.'''
        from tensorflow.python import debug as tf_debug

        # Attach debugger
        self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)

        # Attach Tensorboard debugger
        # self.sess = tf_debug.TensorBoardDebugWrapperSession(
        #         self.sess, "localhost:8080")

    def extend_path(self, p):
        '''Append subdirectory to path `p` named by the model.'''
        if self.prefix:
            return os.path.join(self.prefix, p, self.name)
        return os.path.join(p, self.name)

    def get_summary_outdir(self):
        '''Return the directory where to store the summary.'''
        d = self.extend_path(self.summary_outdir)
        ensure_dir(d)
        return d

    def get_outdir(self):
        '''Return the directory where to store the model.'''
        return self.extend_path(self.outdir)

    def get_plot_path(self):
        '''Return the directory where to store figures.'''
        d = os.path.join(self.get_summary_outdir(), 'plots')
        ensure_dir(d)
        return d

    def clear(self):
        '''Delete summary and model directories.'''
        delete_if_exists(self.get_summary_outdir())
        self.clear_model()

    def denormalize_location(self, items):
        '''Convert normalized carthesian location to true location.'''
        return self.config.denormalize_label(items)

    def clear_model(self):
        '''model directories.'''
        delete_if_exists(self.get_outdir())

    def generate_eval_dataset(self, nrepeat=1):
        '''Generator of evaluation dataset.'''
        return self.config.evaluation_data_generator.get_dataset().batch(
                self.batch_size)

    def generate_predict_dataset(self):
        '''Generator of prediction dataset.'''
        d = self.config.prediction_data_generator
        if not d:
            raise Exception('\nNo prediction data generator defined in config!')
        return d.get_dataset().batch(self.batch_size)

    def generate_dataset(self):
        '''Generator of training dataset.'''
        dataset = self.config.data_generator.get_dataset()
        if self.shuffle_size is not None:
            dataset = dataset.shuffle(buffer_size=self.shuffle_size)
        return dataset.repeat(count=self.n_epochs).batch(self.batch_size)

    def generate_detect_dataset(self):
        '''Generator of prediction dataset.'''
        d = self.config.prediction_data_generator
        if not d:
            raise Exception('\nNo prediction data generator defined in config!')
        return d.get_chunked_dataset(tinc=self.tinc_detect).prefetch(100)

    def model(self, features, labels, mode):
        '''Setup the model.'''
        training = bool(mode == tf.estimator.ModeKeys.TRAIN)

        if self.debug: 
            view = features[:3]
            view = tf.expand_dims(view, -1)
            tf.summary.image('input', view)

        with tf.name_scope('input'):
            input = tf.reshape(
                features, [-1, *self.config.data_generator.tensor_shape,  1])

        input = tf.layers.batch_normalization(input, training=training)
        level = tf.zeros([1], name='level')
                
        for layer in self.layers:
            logger.debug('chain in layer: %s' % layer)
            input, level = layer.chain(input=input, level=level,
                    training=training or self.force_dropout,
                    dropout_rate=self.dropout_rate)

        # Final layer
        predictions = tf.layers.dense(input, self.config.n_classes,
                name='output', activation=None)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode,
                    predictions = {
                        'predictions': self.denormalize_location(predictions),
                        'level': level})

        # do not denormalize labels and predictions for loss
        loss = tf.losses.mean_squared_error(labels, predictions)

        # transform to carthesian coordiantes
        labels = self.denormalize_location(labels)
        predictions = self.denormalize_location(predictions)

        predictions = tf.transpose(predictions) 
        labels = tf.transpose(labels)
        errors = predictions - labels
        abs_errors = tf.abs(errors)
        variable_summaries(errors[0], 'error_abs_x')
        variable_summaries(errors[1], 'error_abs_y')
        variable_summaries(errors[2], 'error_abs_z')

        loss_carthesian = tf.sqrt(tf.reduce_sum(errors ** 2, axis=0,
            keepdims=False))

        variable_summaries(loss_carthesian, 'training_loss')
        loss_ = tf.reduce_mean(loss_carthesian)
        tf.summary.scalar('mean_loss_cart', loss_)
        tf.summary.scalar('loss_normalized', loss)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
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
                    evaluation_hooks=[self.get_summary_hook('eval'),],
                    training_hooks=[
                            self.get_summary_hook('train'), logging_hook])

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
                    evaluation_hooks=[self.get_summary_hook('eval'),
                        # dump_hook
                        ])

    def get_summary_hook(self, subdir=''):
        return tf.train.SummarySaverHook(
            self.summary_nth_step,
            output_dir=pjoin(self.get_summary_outdir(), subdir),
            scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()))

    def train_and_evaluate(self):
        self.save_model_in_summary()
        with self.sess as default:
            self.est = tf.estimator.Estimator(
                model_fn=self.model, model_dir=self.get_outdir())

            train_spec = tf.estimator.TrainSpec(
                    input_fn=self.generate_dataset,
                    max_steps=self.max_steps)

            eval_spec = tf.estimator.EvalSpec(
                    input_fn=self.generate_eval_dataset,
                    steps=None)

            result = tf.estimator.train_and_evaluate(self.est, train_spec, eval_spec)

        self.save_kernels()
        # self.save_activation_maps()

        return result

    def predict(self):
        '''Predict locations prediction data generator and store results
        as a list of carthesian coordinates.'''
        import time
        import csv
        with self.sess as default:
            self.est = tf.estimator.Estimator(
                model_fn=self.model, model_dir=self.get_outdir())
            predictions = []
            for i, p in enumerate(self.est.predict(
                    input_fn=self.generate_predict_dataset,
                    yield_single_examples=False)):

                if not i:
                    tstart = time.time()

                predictions.extend(p['predictions'])

        fn_out = 'predictions.csv'
        with open(fn_out, 'w') as f:
            w = csv.writer(f, delimiter=' ')
            for p in predictions:
                w.writerow(p)

        logger.info('This took %1.1f seconds ' % (time.time()-tstart))
        logger.info('Saved locations in %s' % fn_out)

    def detect(self, tinc=None):
        tpeaksearch = 5.
        detector_threshold = 1.8
        self.tinc_detect = tinc or 1.0
        fn_detector_trace = 'detector.mseed'
        fn_detections = 'detections.pf'

        with self.sess as default:
            self.est = tf.estimator.Estimator(
                model_fn=self.model, model_dir=self.get_outdir())
            detector_level = []

            for ip, p in enumerate(self.est.predict(
                    input_fn=self.generate_detect_dataset,
                    yield_single_examples=True)):
                detector_level.append(p['level'])

            tr = trace.Trace(
                    tmin=self.config.prediction_data_generator.tstart_data,
                    ydata=num.array(detector_level),
                    deltat=self.tinc_detect)

            tpeaks, apeaks = tr.peaks(detector_threshold, tpeaksearch)
            logger.info('Fount %i detections' % len(tpeaks))

            markers = []
            for (tpeak, apeak) in zip(tpeaks, apeaks):
                markers.append(
                        EventMarker(pmodel.Event(time=tpeak,
                    name=str(apeak))))

            logger.info('Saving detections: %s' % fn_detections)
            Marker.save_markers(markers, fn_detections)

            logger.info('Saving detector level: %s' % fn_detector_trace)
            io.save([tr], fn_detector_trace)

    def evaluate_errors(self, n_predict=100):
        logger.debug('evaluation...')
        self.dropout_rate = 0.333
        with self.sess as default:
            self.est = tf.estimator.Estimator(
                model_fn=self.model, model_dir=self.get_outdir())
            labels = [l for _, l in self.config.evaluation_data_generator.generate()]
            labels = self.denormalize_location(num.array(labels))

            all_predictions = []

            # repeating_eval = functools.partialmethod(self.generate_eval_dataset,
            #         nrepeat=n_predict
            for n in range(n_predict):
                predictions = []
                for p in self.est.predict(
                        input_fn=self.generate_eval_dataset,
                        yield_single_examples=False):

                    predictions.append(p['predictions'])
                # predictions = self.denormalize_location(num.array(predictions))
                all_predictions.append(predictions)

                # activate dropout after first iteration. hence, first are
                # 'correct locations'
                self.force_dropout = True

            all_predictions = num.array(all_predictions)

            save_name = pjoin(self.get_plot_path(), 'errors')

            plot.evaluate_errors(
                    all_predictions, labels, name=save_name)

    def evaluate(self, annotate=False):
        logger.debug('evaluation...')

        with self.sess as default:
            self.est = tf.estimator.Estimator(
                model_fn=self.model, model_dir=self.get_outdir())
            labels = [l for _, l in self.config.evaluation_data_generator.generate()]
            predictions = []
            for p in self.est.predict(
                    input_fn=self.generate_eval_dataset,
                    yield_single_examples=False):

                predictions.extend(p['predictions'])

        save_name = pjoin(self.get_plot_path(), 'mislocation')
        predictions = num.array(predictions)
        labels = self.denormalize_location(num.array(labels))

        text_labels = None
        if annotate:
            # text_labels = [str(i) for i in
            #         self.config.evaluation_data_generator.iter_labels()]

            text_labels = self.config.evaluation_data_generator.text_labels

        plot.plot_predictions_and_labels(
                predictions, labels, name=save_name, text_labels=text_labels)

        save_name = pjoin(self.get_plot_path(), 'mislocation_hists')
        plot.mislocation_hist(predictions, labels, name=save_name)

        save_name = pjoin(self.get_plot_path(), 'mislocation_vs_gaps')
        plot.mislocation_vs_gaps(predictions, labels,
                self.config.evaluation_data_generator.gaps(),
                name=save_name)

    def evaluate_station_dropout(self):
        with self.sess as default:
            self.est = tf.estimator.Estimator(
                model_fn=self.model, model_dir=self.get_outdir())
            labels = [l for _, l in self.config.evaluation_data_generator.generate()]
            sdropouts = num.linspace(0, 0.8, 0.1)
            results = {}
            for sdropout in sdropouts:
                predictions = []
                for p in self.est.predict(
                        input_fn=self.generate_eval_dataset,
                        yield_single_examples=False):

                    predictions.extend(p['predictions'])
                results[sdropout] = predictions

        plot.mislocation_vs_gaps_many(results, labels,
                self.config.evaluation_data_generator.gaps(),
                name=save_name)

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
                model_fn=self.model,
                model_dir=self.outdir,
                # params=params,
                config=run_config)

            est.train(input_fn=self.generate_dataset)
            logger.info('====== start evaluation')
            return est.evaluate(input_fn=self.generate_eval_dataset)

    def save_kernels(self, index=0):
        '''save weight kernels of all layers (at index=`index`).'''
        save_path = pjoin(self.get_summary_outdir(), 'kernels')
        ensure_dir(save_path)
        logger.info('Storing weight matrices at %s' % save_path)
        for layer in self.layers:
            layer.visualize_kernel(self.est, save_path=save_path)

    def save_activation_maps(self, index=0):
        save_path = pjoin(self.get_summary_outdir(), 'kernels')
        ensure_dir(save_path)
        for layer in self.layers:
            plot.getActivations(layer, stimuli)

    def save_model_in_summary(self):
        self.regularize()
        self.dump(filename=pjoin(self.get_summary_outdir(), 'model.config'))
