import os
import logging
import tensorflow as tf
import numpy as num

from pyrocko.guts import Object, Float, Int, String, Bool, List, Tuple
from pyrocko.pile import make_pile
from pyrocko.gf.seismosizer import Target

from .data import Noise, Normalization, DataGeneratorBase, Imputation
from .data import ImputationZero, ChannelStackGenerator


logger = logging.getLogger(__name__)


class PinkyConfig(Object):
    '''Configuration of data IO and data preprocessing'''

    blacklist = List.T(
        String.T(), help='List blacklist patterns (may contain wild cards')

    stack_channels = Bool.T(default=False,
        help='If *True* stack abs. amplitudes of all channels of a station')

    sample_length = Float.T(optional=True, help='Length in seconds. Not needed \
        when using TFRecordData')

    data_generator = DataGeneratorBase.T()
    evaluation_data_generator = DataGeneratorBase.T(optional=True)
    prediction_data_generator = DataGeneratorBase.T(optional=True)

    normalization = Normalization.T(default=Normalization(), optional=True)

    absolute = Bool.T(help='Use absolute amplitudes', default=False)

    imputation = Imputation.T(
        optional=True, help='How to mask and fill gaps')
    
    reference_target = Target.T(optional=True)

    n_classes = Int.T(default=3)

    # Not implemented for DataGeneratorBase
    highpass = Float.T(optional=True, help='Highpass filter corner frequency')
    lowpass = Float.T(optional=True, help='Lowpass filter corner frequency')

    highpass_order = Int.T(default=4, optional=True)
    lowpass_order = Int.T(default=4, optional=True)

    normalize_labels = Bool.T(default=True,
        help='Normalize labels by std')

    tpad = Float.T(
        default=0.,
        help='padding between p phase onset and data chunk start')

    t_translation_max = Float.T(default=0.,
        help='Augment data by uniformly shifting examples in time limited by '
        'this parameters. This will increase *tpad*')

    deltat_want = Float.T(optional=True,
        help='If set, down or upsample traces to this sampling rate.')

    # These value or not meant to be modified. If they are set in a
    # configuration this happened automatically to port values accross
    # configurations.
    # _label_scale = num.ones(3, dtype=num.float32, help='(Don\'t modify)')
    # _label_median = num.ones(3, dtype=num.float32, help='(Don\'t modify)')
    _channels =  List.T(
            Tuple.T(4, String.T()), optional=True, help='(Don\'t modify)')
    _n_samples = Int.T(optional=True, help='(Don\'t modify)')

    def setup(self):

        self.data_generator.set_config(self)
        if self.normalize_labels:
            # To not break the label normalization, the data_generator used
            # for training is required in any case at the moment!
            # Better store normalization data during training to recycle at
            # prediction time.
            self._label_median = num.median(
                    num.array(list(
                        self.data_generator.iter_labels())), axis=0)

            self._label_scale = num.mean(num.std(
                    num.array(list(
                        self.data_generator.iter_labels())), axis=0))

        if self.evaluation_data_generator:
            self.evaluation_data_generator.set_config(self)

        if self.prediction_data_generator:
            self.prediction_data_generator.set_config(self)

        self.set_n_samples()

        if self.stack_channels:

            self.data_generator = ChannelStackGenerator.from_generator(
                    generator=self.data_generator)
            if self.evaluation_data_generator:
                self.evaluation_data_generator = ChannelStackGenerator.from_generator(
                    generator=self.evaluation_data_generator)
            if self.prediction_data_generator:
                self.prediction_data_generator = ChannelStackGenerator.from_generator(
                    generator=self.prediction_data_generator)

        # self.data_generator.setup()
        # self.evaluation_data_generator.setup()
        # if self.prediction_data_generator:
        #     self.prediction_data_generator.setup()

    def set_n_samples(self):
        '''Set number of sampes (n_samples) from first example of data
        generator. Note that this assumes that the evaluation data generator
        contains identical shaped examples.'''
        example, _ = next(self.data_generator.generate())
        self._n_samples = example.shape[1]
        assert(example.shape == self.tensor_shape)

    @property
    def effective_deltat(self):
        if self.deltat_want is None:
            return (self.sample_length + self.tpad) / self._n_samples
        else:
            return self.deltat_want

    @property
    def effective_tpad(self):
        tpad = self.tpad + self.t_translation_max
        if self.highpass is not None:
            tpad += 0.5 / self.highpass

        return tpad

    def normalize_label(self, label):
        '''label has to be a numpy array'''
        return (label - self._label_median) / self._label_scale

    def denormalize_label(self, label):
        '''label has to be a numpy array'''
        return (label * self._label_scale) + self._label_median

    @property
    def channels(self):
        return self._channels

    @channels.setter
    def channels(self, v):
        if self._channels:
            logger.warn('Setting channels although channels have been \
                    assigned before')
        self._channels = v

    @property
    def n_channels(self):
        return len(self._channels)

    @property
    def output_shapes(self):
        '''Return a tuple containing the shape of feature arrays and number of
        labels.
        '''
        return (self.tensor_shape, self.n_classes)

    @property
    def tensor_shape(self):
        return (self.n_channels, self._n_samples)
