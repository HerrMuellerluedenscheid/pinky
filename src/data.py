#!/usr/bin/env python3

import tensorflow as tf
from pyrocko.io import save, load
from pyrocko.model import load_stations
from pyrocko.guts import Object, String, Int, Float, Tuple, Bool
from pyrocko.gui import marker
from pyrocko.gf.seismosizer import Engine, Target, LocalEngine
from pyrocko import orthodrome
from pyrocko import pile
from swarm import synthi, source_region

import logging
import random
import numpy as num
import os
import glob
import sys

from .tf_util import _FloatFeature, _Int64Feature, _BytesFeature

pjoin = os.path.join
EPSILON = 1E-4


# TODOS:
# - remove 'double events'

class Noise(Object):
    level = Float.T(default=1.)

    def __init__(self, *args, **kwargs):
        super(Noise, self).__init__(*args, **kwargs)
        logging.info('Applying noise to input data with level %s' % self.level)

    def get_chunk(self, n_channels, n_samples):
        ...


class WhiteNoise(Noise):

    def get_chunk(self, n_channels, n_samples):
        return num.random.random((n_channels, n_samples)).astype(num.float32) * self.level


class DataGeneratorBase(Object):
    _shape = Tuple.T(2, Int.T(), optional=True)
    fn_tfrecord = String.T(optional=True)
    n_classes = Int.T(default=3)

    @property
    def tensor_shape(self):
        return self._shape

    @property
    def generate_output_types(self):
        return tf.float32, tf.float32

    @tensor_shape.setter
    def tensor_shape(self, v):
        if v == self._shape:
            return self._shape
        else:
            self._shape = v

    @property
    def output_shapes(self):
        return (self.tensor_shape, self.n_classes)

    def unpack_examples(self, record_iterator):
        '''Parse examples stored in TFRecordData to `tf.train.Example`'''
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            chunk = example.features.feature['data'].bytes_list.value[0]
            label = example.features.feature['label'].bytes_list.value[0]

            chunk = num.fromstring(chunk, dtype=num.float32)
            chunk = chunk.reshape(self.tensor_shape)
            label = num.fromstring(label, dtype=num.float32)
            yield chunk, label

    def generate(self):
        record_iterator = tf.python_io.tf_record_iterator(
                path=self.fn_tfrecord)
        return self.unpack_examples(record_iterator)

    def get_dataset(self):
        return tf.data.Dataset.from_generator(
            self.generate,
            self.generate_output_types,
            output_shapes=self.output_shapes)

    def pack_examples(self, generator):
        '''Serialize Examples to strings.'''
        for ydata, label in self.generate():
            yield tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'data': _BytesFeature(ydata.tobytes()),
                        'label': _BytesFeature(num.array(label, dtype=num.float32).tobytes()),
                    }))

    def write(self, directory):
        '''Write example data to TFRecordDataset using `self.writer`.'''
        writer = tf.python_io.TFRecordWriter(directory)
        for ex in self.pack_examples(writer):
            writer.write(ex.SerializeToString())


class DataGenerator(DataGeneratorBase):

    sample_length = Float.T(help='length [s] of data window')
    fn_stations = String.T()
    absolute = Bool.T(help='Use absolute amplitudes', default=False)
    effective_deltat = Float.T(optional=True)
    reference_target = Target.T(
        default=Target(
            codes=('', 'NKC', '', 'SHZ'),
            lat=50.2331,
            lon=12.448,
            elevation=546))
    noise = Noise.T(optional=True, help='Add noise to your feature chunks')

    highpass = Float.T(optional=True)
    lowpass = Float.T(optional=True)

    highpass_order = Int.T(default=4, optional=True)
    lowpass_order = Int.T(default=4, optional=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setup()

    def setup(self):
        pass

    @property
    def output_shapes(self):
        return (self.tensor_shape, self.n_classes)

    def extract_labels(self, source):
        n, e = orthodrome.latlon_to_ne(
            self.reference_target.lat, self.reference_target.lon,
            source.lat, source.lon)
        return (n, e, source.depth)

    def get_raw_data_chunk(self):
        '''
        Return an array of size (Nchannels x Nsamples_max).

        When working with noisy data, replace this function.
        '''
        return num.zeros(self.tensor_shape, dtype=num.float32)

    def attach_graph(self, dataset, shape):
        '''
        Use this method to attach any preprocessing to be done in tensorflow
        graph.
        '''
        return dataset

    def regularize_deltat(self, tr):
        '''Equalize sampling rates accross the data set according to sampling rate
        set in `self.config`.'''
        if abs(tr.deltat - self.effective_deltat)>0.0001:
            tr.resample(self.effective_deltat)

    def fit_data_into_chunk(self, traces, chunk, indices=None, tref=0):
        indices = indices or range(len(traces))
        for i, tr in zip(indices, traces):

            if self.absolute:
                tr.data = num.abs(tr.data)

            data_len = len(tr.data)
            istart_trace = int((tr.tmin - tref) / tr.deltat)
            istart_array = max(istart_trace, 0)
            istart_trace = max(-istart_trace, 0)
            istop_array = istart_array + (data_len - 2* istart_trace)

            ydata = tr.data[ \
                istart_trace: min(data_len, self.n_samples_max-istart_array) \
                + istart_trace]
            chunk[i, istart_array: istart_array+ydata.shape[0]] += ydata

        chunk -= num.min(chunk)
        chunk /= num.max(chunk)


class PileData(DataGenerator):
    '''Data generator for locally saved data.'''
    data_path = String.T()
    data_format = String.T(default='mseed')
    fn_markers = String.T()
    deltat_want = Float.T(optional=True)
    shuffle = Bool.T(default=False)

    def setup(self):
        self.data_pile = pile.make_pile(
                self.data_path, fileformat=self.data_format)

        if self.data_pile.is_empty():
            sys.exit('Data pile is empty!')

        self.deltat_want = self.deltat_want or \
                min(self.data_pile.deltats.keys())
        self.n_samples_max = int(self.sample_length/self.deltat_want)
        logging.debug('loading markers')

        markers = marker.load_markers(self.fn_markers)
        marker.associate_phases_to_events(markers)

        markers_by_nsl = {}
        for m in markers:
            if not m.match_nsl(self.reference_target.codes[:3]):
                continue
            key = m.one_nslc()[:3]
            _ms = markers_by_nsl.get(key, [])
            _ms.append(m)
            markers_by_nsl[key] = _ms

        assert(len(markers_by_nsl) == 1)
        self.markers = list(markers_by_nsl.values())[0]

        if self.shuffle:
            random.shuffle(self.markers)
        else:
            self.markers.sort(key=lambda x: x.tmin)

        self.channels = list(self.data_pile.nslc_ids.keys())
        self.tensor_shape = (len(self.channels), self.n_samples_max)

    def check_inputs(self):
        if len(self.data_pile.deltats()) > 1:
            logging.warn(
                'Different sampling rates in dataset. Preprocessing slow')

    def preprocess(self, tr):
        '''Trace preprocessing

        Ensures equal sampling rates
        
        :param tr: pyrocko.tr.Trace object'''
        if tr.delta - self.deltat_want > EPSILON:
            tr.resample(self.deltat_want)
        elif tr.deltat - self.deltat_want < -EPSILON:
            tr.downsample_to(self.deltat_want)

    def generate(self):
        tr_len = self.n_samples_max * self.deltat_want
        nslc_to_index = {nslc: idx for idx, nslc in enumerate(self.channels)}
        for m in self.markers:
            event = m.get_event()
            if event is None:
                logging.debug('No event: %s' % m)
                continue

            for trs in self.data_pile.chopper(
                    tmin=m.tmin, tmax=m.tmin+tr_len, keep_current_files_open=True):

                if not len(trs):
                    logging.debug('No data at tmin=%s' % m.tmin)
                    continue

                for tr in trs:
                    tr.data = tr.ydata.astype(num.float)

                chunk = self.get_raw_data_chunk()
                indices = [nslc_to_index[tr.nslc_id] for tr in trs]
                self.fit_data_into_chunk(trs, chunk, indices=indices, tref=m.tmin)

                if self.noise is not None:
                    chunk += self.noise.get_chunk(*self.tensor_shape)

                yield chunk, self.extract_labels(event)


class GFSwarmData(DataGenerator):
    swarm = source_region.Swarm.T()
    n_sources = Int.T(default=100)
    onset_phase = String.T(default='p')
    quantity = String.T(default='velocity')
    tpad = Float.T(default=1., help='padding between p phase onset and data chunk start')

    def setup(self):
        stations = load_stations(self.fn_stations)
        self.targets = synthi.guess_targets_from_stations(
            stations, quantity=self.quantity)
        self.store = self.swarm.engine.get_store()  # uses default store
        self.n_samples_max = int(self.sample_length/self.store.config.deltat)
        self.tensor_shape = (len(self.targets), self.n_samples_max)

    def make_data_chunk(self, source, results):
        ydata_stacked = self.get_raw_data_chunk()
        tref = self.store.t(
            self.onset_phase, (
                source.depth,
                source.distance_to(self.reference_target))
            )

        tref += (source.time - self.tpad)
        traces = [result.trace for result in results]
        self.fit_data_into_chunk(traces, ydata_stacked, tref=tref)

        return ydata_stacked

    def extract_labels(self, source):
        elat, elon = source.effective_latlon
        n, e = orthodrome.latlon_to_ne(
            self.reference_target.lat, self.reference_target.lon,
            elat, elon)
        return (n, e, source.depth)

    def generate(self):
        sources = self.swarm.get_effective_sources()
        self.tensor_shape = (len(self.targets), self.n_samples_max)

        response = self.swarm.engine.process(
            sources=sources,
            targets=self.targets)

        for isource, source in enumerate(response.request.sources):
            chunk = self.make_data_chunk(source, response.results_list[isource])

            if self.noise is not None:
                chunk += self.noise.get_chunk(*self.tensor_shape)

            yield chunk, self.extract_labels(source)

    @classmethod
    def get_example(cls):
        gf_engine = LocalEngine(
            use_config=True,
            store_superdirs=['/data/stores'],
            # default_store_id='test_store'
        )

        example_swarm = source_region.Swarm(
            geometry=source_region.CuboidSourceGeometry(),
            timing=source_region.RandomTiming(),
            engine=gf_engine
        )

        return cls(
            fn_stations='stations.pf', sample_length=10, swarm=example_swarm,
        )

