import tensorflow as tf
import os
import logging
from pyrocko.guts import Object, Float, Int, String, Bool, List, Tuple

from pyrocko.pile import make_pile
from pyrocko.gf.seismosizer import Target

from .data import Noise, Normalization, DataGeneratorBase, Imputation
from .data import ImputationZero, ChannelStackGenerator


logger = logging.getLogger(__name__)


class PinkyConfig(Object):

    blacklist = List.T(
        String.T(), help='List blacklist patterns (may contain wild cards')

    stack_channels = Bool.T(default=False)
    sample_length = Float.T(optional=True, help='Length in seconds. Not needed \
        when using TFRecordData')

    data_generator = DataGeneratorBase.T()
    evaluation_data_generator = DataGeneratorBase.T()
    prediction_data_generator = DataGeneratorBase.T(optional=True)
    normalization = Normalization.T(default=Normalization(), optional=True)
    imputation = Imputation.T(
        optional=True, help='How to mask and fill gaps')
    
    _n_samples = Int.T(optional=True)
    reference_target = Target.T(optional=True)

    n_classes = Int.T(default=3)
    _channels =  List.T(
            Tuple.T(4, String.T()), optional=True, help='(Don\'t modify)')

    # Not implemented for DataGeneratorBase
    highpass = Float.T(optional=True, help='highpass filter corner frequency')
    lowpass = Float.T(optional=True, help='lowpass filter corner frequency')

    highpass_order = Int.T(default=4, optional=True)
    lowpass_order = Int.T(default=4, optional=True)

    absolute = Bool.T(help='Use absolute amplitudes', default=False)
    tpad = Float.T(default=0.,
            help='padding between p phase onset and data chunk start')
    deltat_want = Float.T(optional=True)

    def setup(self):
        self.data_generator.set_config(self)

        self.evaluation_data_generator.set_config(self)

        if self.prediction_data_generator:
            self.prediction_data_generator.set_config(self)

        self.set_n_samples()

        if self.stack_channels:
            self.data_generator = ChannelStackGenerator.from_generator(
                    generator=self.data_generator)
            self.evaluation_data_generator = ChannelStackGenerator.from_generator(
                    generator=self.evaluation_data_generator)

        self.data_generator.setup()
        self.evaluation_data_generator.setup()

    def set_n_samples(self):
        '''Set number of sampes (n_samples) from first example of data
        generator. Note that this assumes that the evaluation data generator
        contains identical shaped examples.'''
        example, _ = next(self.data_generator.generate())
        self._n_samples = example.shape[1]
        assert(example.shape == self.tensor_shape)

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
