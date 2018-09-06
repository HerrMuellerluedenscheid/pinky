#!/usr/bin/env python3

import tensorflow as tf
from pyrocko.io import save, load
from pyrocko.model import load_stations
from pyrocko import guts
from pyrocko.guts import Object, String, Int, Float, Tuple, Bool, Dict, List
from pyrocko.gui import marker
from pyrocko.gf.seismosizer import Engine, Target, LocalEngine, Source
from pyrocko import orthodrome
from pyrocko import pile
from swarm import synthi, source_region
from collections import defaultdict, OrderedDict
from functools import lru_cache
from pyrocko import util

import logging
import random
import numpy as num
import os
import glob
import sys

from .tf_util import _FloatFeature, _Int64Feature, _BytesFeature
from .util import delete_if_exists, first_element, filter_oob


pjoin = os.path.join
EPSILON = 1E-4


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Normalization(Object):
    def __call__(self, chunk):
        pass


class NormalizeMax(Normalization):
    '''Normalize the entire chunk by chunk's absolut maximum.'''
    
    def __call__(self, chunk):
        chunk /= num.nanmax(num.abs(chunk))


class NormalizeLog(Normalization):
    def __call__(self, chunk):
        mean = num.nanmean(chunk)
        chunk -= mean 
        sign = num.sign(chunk)

        chunk[:] = num.log(num.abs(chunk)+EPSILON)
        chunk *= sign
        chunk += mean


class NormalizeNthRoot(Normalization):
    '''Normalize traces by their Nth root.
    
    Note: polarities get lost.'''
    nth_root = Int.T(default=4)

    def __call__(self, chunk):
        chunk -= num.nanmean(chunk)
        chunk[:] = num.power(num.abs(chunk), 1./self.nth_root)
        chunk += num.nanmean(chunk)


class NormalizeChannelMax(Normalization):
    def __call__(self, chunk):
        trace_levels = num.nanmean(chunk, axis=1)[:, num.newaxis]
        chunk -= trace_levels
        chunk /= (num.nanmax(num.abs(chunk), axis=1)[:, num.newaxis]) * 2.
        chunk += trace_levels


class NormalizeStd(Normalization):
    '''Normalizes by dividing through the standard deviation'''
    def __call__(self, chunk):
        # save and subtract trace levels
        # trace_levels = num.nanmean(chunk, axis=1)
        trace_levels = num.nanmean(chunk)
        chunk -= trace_levels

        # normalize
        # chunk /= num.std(chunk, axis=1)
        chunk /= num.std(chunk)

        # restore mean levels
        chunk += trace_levels


class NormalizeChannelStd(Normalization):
    '''Normalizes by dividing through the standard deviation'''
    def __call__(self, chunk):
        # save and subtract trace levels
        trace_levels = num.nanmean(chunk, axis=1)[:, num.newaxis]
        chunk -= trace_levels

        # normalize
        nanstd = num.nanstd(chunk, axis=1)[:, num.newaxis]
        nanstd[nanstd==0] = 1.
        chunk /= nanstd

        # restore mean levels
        chunk += trace_levels


class Noise(Object):
    level = Float.T(default=1., optional=True)

    def __call__(self, chunk):
        pass


class WhiteNoise(Noise):

    def __call__(self, chunk):
        chunk += num.random.random(chunk.shape) * self.level


class Imputation(Object):

    def __call__(self, *args, **kwargs):
        pass


class ImputationZero(Imputation):
    def __call__(self, chunk):
        return 0.


class ImputationMean(Imputation):

    def __call__(self, chunk):
        return num.nanmean(chunk).astype(num.float32)


class DataGeneratorBase(Object):
    '''This is the base class for all generators.
    
    This class to dump and load data to/from all subclasses into
    TFRecordDatasets.
    '''
    _shape = Tuple.T(2, Int.T(), optional=True, help='(Don\'t modify)')
    _channels =  List.T(Tuple.T(4, String.T()), optional=True, help='(Don\'t modify)')
    fn_tfrecord = String.T(optional=True)
    n_classes = Int.T(default=3)
    noise = Noise.T(default=Noise(), help='Add noise to feature')
    normalization = Normalization.T(default=NormalizeMax(), optional=True)

    station_dropout_rate = Float.T(default=0.,
        help='Rate by which to mask all channels of station')

    imputation = Imputation.T(
        default=ImputationZero(),
        help='How to mask and fill gaps')

    blacklist = List.T(
        String.T(), help='List blacklist patterns (may contain wild cards')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setup()

    def setup(self):
        pass

    @property
    def channels(self):
        return self._channels

    @channels.setter
    def channels(self, v):
        self._channels = v

    @property
    @lru_cache(maxsize=1)
    def nsl_to_indices(self):
        ''' Returns a dictionary which maps nsl codes to indexing arrays.'''
        indices = defaultdict(list)
        for nslc, index in self.nslc_to_index.items():
            indices[nslc[:3]].append(index)

        for k in indices.keys():
            indices[k] = num.array(indices[k])

        return indices

    @property
    @lru_cache(maxsize=1)
    def nsl_indices(self):
        ''' Returns a 2D array of indices of channels belonging to one station.'''
        return [v for v in self.nsl_to_indices.values()]

    @property
    def nslc_to_index(self):
        ''' Returns a dictionary which maps nslc codes to trace indices.'''
        d = OrderedDict()
        idx = 0
        for nslc in self.channels:
            if not util.match_nslc(self.blacklist, nslc):
                d[nslc] = idx
                idx += 1
        return d
        # items = ['%s : %s' % ('.'.join(k), v) for k, v in d.items()]
        # logger.debug('Setting nslc-to-index mapping:\n')
        # logger.debug('\n'.join(items))
        # return d

    def reject_blacklisted(self, tr):
        '''returns `False` if nslc codes of `tr` match any of the blacklisting
        patters. Otherwise returns `True`'''
        return not util.match_nslc(self.blacklist, tr.nslc_id)

    @property
    def tensor_shape(self):
        return self._shape

    @property
    def generate_output_types(self):
        '''Return data types of features and labels'''
        return tf.float32, tf.float32

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

    def iter_labels(self):
        '''Iterate through labels.'''
        for _, label in self.generate():
            yield label

    def generate(self):
        record_iterator = tf.python_io.tf_record_iterator(
                path=self.fn_tfrecord)
        return self.unpack_examples(record_iterator)

    def get_dataset(self):
        return tf.data.Dataset.from_generator(
            self.generate,
            self.generate_output_types,
            output_shapes=self.output_shapes)

    def get_raw_data_chunk(self):
        '''Return an array of size (Nchannels x Nsamples_max) filled with
        NANs.'''
        empty_array = num.empty(self.tensor_shape, dtype=num.float32)
        empty_array.fill(num.nan)
        return empty_array

    def pack_examples(self):
        '''Serialize Examples to strings.'''
        for ydata, label in self.generate():
            yield tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'data': _BytesFeature(ydata.tobytes()),
                        'label': _BytesFeature(num.array(
                            label, dtype=num.float32).tobytes()),
                    }))

    def mask(self, chunk):
        '''For data augmentation: Mask traces in chunks'''
        indices = self.nsl_indices
        a = num.random.random(len(indices))
        i = num.where(a < self.station_dropout_rate)[0]
        imputation_value = self.imputation(chunk)
        for ii in i:
            chunk[indices[ii], :] = imputation_value

    def process_chunk(self, chunk):
        '''Probably better move this to the tensorflow side for better
        performance.'''

        # fill gaps
        chunk[num.isnan(chunk)] = self.imputation(chunk)

        # mask data
        self.mask(chunk)

        # add noise
        self.noise(chunk)

        # apply normalization
        self.normalization(chunk)

        return chunk

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
            yield self.process_chunk(chunk), label

    def write(self, directory):
        '''Write example data to TFRecordDataset using `self.writer`.'''
        logger.debug('writing TFRecordDataset: %s' % directory)
        writer = tf.python_io.TFRecordWriter(directory)
        for ex in self.pack_examples():
            writer.write(ex.SerializeToString())

    def cleanup(self):
        '''Remove remaining folders'''
        delete_if_exists(self.fn_tfrecord)


class ChannelStackGenerator(DataGeneratorBase):
    '''Stack summed absolute traces of all available channels of a station
    provided by the `generator`.
    '''
    generator = DataGeneratorBase.T(help='The generator to be compressed')

    def setup(self):
        self.channels = [k + ('STACK', ) for k in
                self.generator.nsl_to_indices.keys()]
        self.tensor_shape = (len(self.channels), self.generator.tensor_shape[1])

    def generate(self):
        index_mapping_in = self.generator.nsl_to_indices

        d = OrderedDict()
        for idx, nsl in enumerate(self.channels):
            d[nsl[:3]] = idx

        for feature, label in self.generator.generate():
            chunk = self.get_raw_data_chunk()
            for nsl, indices in index_mapping_in.items():
                chunk[d[nsl]] = num.sum(num.abs(feature[indices, :]), axis=0)
            yield chunk, label


class DataGenerator(DataGeneratorBase):

    '''Generate examples from data on hard drives.'''
    sample_length = Float.T(help='length [s] of data window')
    absolute = Bool.T(help='Use absolute amplitudes', default=False)
    effective_deltat = Float.T(optional=True)
    tpad = Float.T(default=0.,
            help='padding between p phase onset and data chunk start')
    deltat_want = Float.T(optional=True)
    reference_target = Target.T(
        default=Target(
            codes=('', 'NKC', '', 'SHZ'),
            lat=50.2331,
            lon=12.448,
            elevation=546))
    highpass = Float.T(optional=True)
    lowpass = Float.T(optional=True)

    highpass_order = Int.T(default=4, optional=True)
    lowpass_order = Int.T(default=4, optional=True)

    def preprocess(self, tr):
        '''Trace preprocessing

        :param tr: pyrocko.tr.Trace object
        
        Demean, type casting to float32, filtering, adjust sampling rates.
        '''
        tr.ydata = tr.ydata.astype(num.float32)
        tr.ydata -= num.mean(tr.ydata)

        filter_function = self.get_filter_function()
        filter_function(tr)

        if self.deltat_want is not None:
            if tr.deltat - self.deltat_want > EPSILON:
                tr.resample(self.deltat_want)
            elif tr.deltat - self.deltat_want < -EPSILON:
                tr.downsample_to(self.deltat_want)

    @lru_cache(maxsize=1)
    def get_filter_function(self):
        '''Setup and return a function that takes applies high- and lowpass
        filters to :py:class:`pyrocko.trace.Trace` instances.
        '''
        functions = []
        if self.highpass is not None:
            functions.append(lambda tr: tr.highpass(
                corner=self.highpass,
                order=self.highpass_order))
        if self.lowpass is not None:
            functions.append(lambda tr: tr.lowpass(
                corner=self.lowpass,
                order=self.lowpass_order))

        def fn(t):
            for f in functions:
                f(t)
            if self.absolute:
                t.ydata = num.abs(t.ydata)

        return fn

    def extract_labels(self, source):
        n, e = orthodrome.latlon_to_ne(
            self.reference_target.lat, self.reference_target.lon,
            source.lat, source.lon)
        return (n, e, source.depth)

    def fit_data_into_chunk(self, traces, indices=None, tref=0):
        '''Fit all `traces` into a 2 demensional numpy array.
        
        :param traces: list of pyrocko.trace.Trace instances
        :param indices: list of indices where the traces in `traces` are
            supposed to be filled into the array. If this parameter is None
            assumes that the order of traces will remain the same accross
            iterations.
        :param tref: absolute time where to chop the data chunk. Typically first
            p phase onset'''
        chunk = self.get_raw_data_chunk()
        indices = indices or range(len(traces))
        tref = tref - self.tpad
        for i, tr in zip(indices, traces):
            data_len = len(tr.ydata)
            istart_trace = int((tr.tmin - tref) / tr.deltat)
            istart_array = max(istart_trace, 0)
            istart_trace = max(-istart_trace, 0)
            istop_array = istart_array + (data_len - 2* istart_trace)

            ydata = tr.ydata[
                istart_trace: min(data_len, self.n_samples_max-istart_array) \
                    + istart_trace]
            chunk[i, istart_array: istart_array+ydata.shape[0]] = ydata

        i_mask = num.isnan(chunk)
        i_unmask = num.logical_not(i_mask)
        chunk[i_unmask] /= (num.nanmax(num.abs(chunk)) * 2.)
        chunk[i_unmask] += 0.5
        return chunk


class PileData(DataGenerator):
    '''Data generator for locally saved data.'''
    fn_stations = String.T()
    data_paths = List.T(String.T())
    data_format = String.T(default='detect')
    fn_markers = String.T()
    sort_markers = Bool.T(default=False,
            help= 'Sorting markers speeds up data io. Shuffled markers \
            improve generalization')


    def setup(self):
        self.data_pile = pile.make_pile(
            self.data_paths, fileformat=self.data_format)

        if self.data_pile.is_empty():
            sys.exit('Data pile is empty!')

        self.deltat_want = self.deltat_want or \
                min(self.data_pile.deltats.keys())
        self.n_samples_max = int(
                (self.sample_length + self.tpad) / self.deltat_want)

        logger.debug('loading markers')
        markers = marker.load_markers(self.fn_markers)

        if self.sort_markers:
            logger.info('sorting markers!')
            markers.sort(key=lambda x: x.tmin)

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

        # filter markers that do not have an event assigned:
        self.markers = list(markers_by_nsl.values())[0]
        self.markers = [m for m in self.markers if m.get_event() is not None]

        self.channels = list(self.data_pile.nslc_ids.keys())
        self.channels.sort()
        self.tensor_shape = (len(self.channels), self.n_samples_max)

    def check_inputs(self):
        if len(self.data_pile.deltats()) > 1:
            logger.warn(
                'Different sampling rates in dataset. Preprocessing slow')

    def extract_labels(self, marker):
        source = marker.get_event()
        n, e = orthodrome.latlon_to_ne(
            self.reference_target.lat, self.reference_target.lon,
            source.lat, source.lon)
        return (n, e, source.depth)

    def iter_labels(self):
        for m in self.markers:
            yield self.extract_labels(m)

    def generate(self):
        tr_len = self.n_samples_max * self.deltat_want
        nslc_to_index = self.nslc_to_index

        tpad = self.tpad
        if self.highpass is not None:
            tpad += 0.5 / self.highpass

        for i_m, m in enumerate(self.markers):
            logger.debug('processig marker %s / %s' % (i_m, len(self.markers)))

            for trs in self.data_pile.chopper(
                    tmin=m.tmin-tpad, tmax=m.tmax+tr_len+tpad,
                    keep_current_files_open=True,
                    want_incomplete=False,
                    trace_selector=self.reject_blacklisted):

                for tr in trs:
                    self.preprocess(tr)

                indices = [nslc_to_index[tr.nslc_id] for tr in trs]
                chunk = self.fit_data_into_chunk(
                        trs, indices=indices, tref=m.tmin)

                yield self.process_chunk(chunk), self.extract_labels(m)



class GFSwarmData(DataGenerator):
    fn_stations = String.T()
    swarm = source_region.Swarm.T()
    n_sources = Int.T(default=100)
    onset_phase = String.T(default='first(p|P)')
    quantity = String.T(default='velocity')

    def setup(self):
        stations = load_stations(self.fn_stations)
        self.targets = synthi.guess_targets_from_stations(
            stations, quantity=self.quantity)
        self.store = self.swarm.engine.get_store()  # uses default store
        self.n_samples_max = int((self.sample_length + self.tpad) / self.store.config.deltat)
        self.tensor_shape = (len(self.targets), self.n_samples_max)

    def make_data_chunk(self, source, results):
        tref = self.store.t(
            self.onset_phase, (
                source.depth,
                source.distance_to(self.reference_target))
            )

        # what about tref at other locations
        tref += (source.time - self.tpad)
        traces = [result.trace for result in results]
        ydata_stacked = self.fit_data_into_chunk(traces, tref=tref)

        return ydata_stacked

    def extract_labels(self, source):
        elat, elon = source.effective_latlon
        n, e = orthodrome.latlon_to_ne(
            self.reference_target.lat, self.reference_target.lon,
            elat, elon)
        return (n, e, source.depth)

    def generate(self):
        response = self.swarm.engine.process(
            sources=self.swarm.get_effective_sources(),
            targets=self.targets)

        for isource, source in enumerate(response.request.sources):
            chunk = self.make_data_chunk(source, response.results_list[isource])
            yield self.process_chunk(chunk), self.extract_labels(source)

    @classmethod
    def get_example(cls):
        gf_engine = LocalEngine(
            use_config=True,
            store_superdirs=['/data/stores'],
            default_store_id='test_store'
        )

        example_swarm = source_region.Swarm(
            geometry=source_region.CuboidSourceGeometry(),
            timing=source_region.RandomTiming(),
            engine=gf_engine
        )

        return cls(
            fn_stations='stations.pf', sample_length=10, swarm=example_swarm,
        )

        
class SeismosizerData(DataGenerator):
    fn_sources = String.T(
            help='filename containing pyrocko.gf.seismosizer.Source instances')
    fn_targets = String.T(
            help='filename containing pyrocko.gf.seismosizer.Target instances')
    engine = LocalEngine.T()
    onset_phase = String.T(default='first(p|P)')

    def setup(self):
        self.sources = guts.load(filename=self.fn_sources)
        self.targets = guts.load(filename=self.fn_targets)
        self.channels = [t.codes for t in self.targets]
        store_ids = [t.store_id for t in self.targets]
        store_id = set(store_ids)
        assert len(store_id) == 1, 'More than one store used. Not \
                implemented yet'
        self.store = self.engine.get_store(store_id.pop())

        filter_oob(self.sources, self.targets, self.store.config)

        dt = self.deltat_want or self.store.config.deltat
        self.n_samples_max = int((self.sample_length + self.tpad) / dt)
        self.tensor_shape = (len(self.targets), self.n_samples_max)

    def extract_labels(self, source):
        # print('labels n%s, e%s, z%s' % ( source.north_shift, source.east_shift,
        # source.depth))
        return (source.north_shift, source.east_shift, source.depth)

    def generate(self):
        response = self.engine.process(
            sources=self.sources,
            targets=self.targets)

        for isource, source in enumerate(response.request.sources):
            traces = [x.trace.pyrocko_trace() for x in \
                    response.results_list[isource]]
            
            for tr in traces:
                self.preprocess(tr)
            arrivals = [self.store.t(self.onset_phase,
                (source.depth, source.distance_to(t))) for t in self.targets]
            tref = min([a for a in arrivals if a is not None])
            chunk = self.fit_data_into_chunk(traces, tref=tref+source.time)
            yield self.process_chunk(chunk), self.extract_labels(source)
