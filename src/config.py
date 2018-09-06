import tensorflow as tf
from pyrocko.guts import Object, Float, Int, String, Bool

from pyrocko.pile import make_pile
# from .data import DataGeneratorBase
from .data import *
import os


class PinkyConfig(Object):
    noise = Noise.T(default=Noise(), help='Add noise to feature')
    normalization = Normalization.T(default=NormalizeMax(), optional=True)
    station_dropout_rate = Float.T(default=0.,
        help='Rate by which to mask all channels of station')
    imputation = Imputation.T(default=ImputationZero(), help='How to mask and fill \
        gaps (options: zero | mean)')

    blacklist = List.T(
        String.T(), help='List blacklist patterns (may contain wild cards')

    stack_channels = Bool.T()
    sample_length = Float.T()


class PremadeGeneratorConfig(Object):
    window_length = Float.T(default=5.)
    data_noise = String.T()
    data_events = String.T()
    out_fn = String.T()
    target_deltat = Int.T(optional=True)
    ref_lat = Float.T()
    ref_lon = Float.T()
    batch_size = Int.T()
    n_classes = Int.T()

    def __init__(self, *args, **kwargs):
        Object.__init__(self, *args, **kwargs)
        self._writer = None
        self._p1 = None
        self._p2 = None
        self._effective_deltat = None

    def load_data(self):
        if None in (self._p1, self._p2):
            print('load data... should happen just once!')
            self._p1 = make_pile(self.data_noise)
            self._p2 = make_pile(self.data_events)
            if (len(self._p1.deltats) > 1 or len(self._p2.deltats)>1) and self.target_deltat is None:
                raise Exception('WARNING : target_deltat not defined and different sampling rates')
        return self._p1, self._p2

    @property
    def effective_deltat(self):
        if self._effective_deltat is None:
            if self.target_deltat is not None:
                self._effective_deltat = self.target_deltat
            else:
                self._effective_deltat = list(self._p1.deltats.keys())[0]
        return self._effective_deltat

    @property
    def nslc_ids(self):
        p1, p2 = self.load_data()
        k1 = tuple(p1.nslc_ids.keys())
        k2 = tuple(p2.nslc_ids.keys())
        ids = list(set(k1 + k2))
        return ids

    @property
    def writer(self):
        if self._writer is None:
            outpath = self.out_fn.rsplit('/', 1)[0]
            if not os.path.exists(outpath):
                os.makedirs(outpath)                

            self._writer = tf.python_io.TFRecordWriter(self.out_fn)
        return self._writer



