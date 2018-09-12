import tensorflow as tf
from pyrocko.guts import Object, Float, Int, String, Bool, List, Tuple

from pyrocko.pile import make_pile
from pyrocko.gf.seismosizer import Target

from .data import Noise, Normalization, DataGeneratorBase, Imputation
from .data import ImputationZero, ChannelStackGenerator
import os


class PinkyConfig(Object):

    blacklist = List.T(
        String.T(), help='List blacklist patterns (may contain wild cards')

    stack_channels = Bool.T(default=False)
    sample_length = Float.T(optional=True, help='Length in seconds. Not needed \
        when using TFRecordData')

    data_generator = DataGeneratorBase.T()
    evaluation_data_generator = DataGeneratorBase.T()
    _shape = Tuple.T(2, Int.T(), optional=True, help='(Don\'t modify)')
    _channels =  List.T(Tuple.T(4, String.T()), optional=True, help='(Don\'t modify)')

    normalization = Normalization.T(default=Normalization(), optional=True)

    imputation = Imputation.T(
        default=ImputationZero(),
        optional=True,
        help='How to mask and fill gaps')
    
    reference_target = Target.T(optional=True)

    n_classes = Int.T(default=3)

    def setup(self):
        self.data_generator.set_config(self)
        self.evaluation_data_generator.set_config(self)

        if self.stack_channels:
            self.data_generator = ChannelStackGenerator.from_generator(
                    generator=self.data_generator)
            self.evaluation_data_generator = ChannelStackGenerator.from_generator(
                    generator=self.evaluation_data_generator)

        # self.data_generator.set_config(self)
        # self.evaluation_data_generator.set_config(self)
        self.data_generator.setup()
        self.evaluation_data_generator.setup()

    @property
    def tensor_shape(self):
        return self._shape

    @tensor_shape.setter
    def tensor_shape(self, v):
        if v == self._shape:
            return self._shape
        else:
            self._shape = v

    @property
    def output_shapes(self):
        '''Return a tuple containing the shape of feature arrays and number of
        labels.
        '''
        return (self.tensor_shape, self.n_classes)

