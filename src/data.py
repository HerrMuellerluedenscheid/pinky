#!/usr/bin/env python3

import tensorflow as tf
from pyrocko.io import save, load
from pyrocko.model import load_stations, load_events, Event
from pyrocko import guts
from pyrocko.guts import Object, String, Int, Float, Tuple, Bool, Dict, List
from pyrocko.gui import marker
from pyrocko.gf.seismosizer import Engine, Target, LocalEngine, Source
from pyrocko import orthodrome
from pyrocko import pile

from collections import defaultdict, OrderedDict
from functools import lru_cache, partialmethod, partial
from pyrocko import util

import logging
import random
import numpy as num
import os
import glob
import sys
import copy

from .tf_util import _FloatFeature, _Int64Feature, _BytesFeature
from .util import delete_if_exists, first_element, filter_oob, ensure_list, snr


pjoin = os.path.join
EPSILON = 1E-9
UNLABELED = (-9E9, -9E9, -9E9)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def all_NAN(d):
    return num.all(num.isnan(d))


class ChunkOperation(Object):
    '''Modifies a data chunk (2D image) when called.'''

    def __call__(self, chunk):
        pass


class Normalization(ChunkOperation):
    '''Data normalization in 2D'''
    pass
    


class NormalizeMax(Normalization):
    '''Normalize the entire chunk by chunk's absolut maximum.'''

    def __call__(self, chunk):
        chunk /= num.nanmax(num.abs(chunk))


class NormalizeLog(Normalization):
    '''Normalize by taking the logarithm of amplitudes.'''
    def __call__(self, chunk):
        chunk -= num.nanmedian(chunk)
        # sign = num.sign(chunk)

        chunk[:] = num.log(num.abs(chunk)+EPSILON)
        # chunk /= num.std(chunk)
        # chunk /= num.nanmax(chunk)
        # chunk *= sign
        # chunk += mean


class NormalizeFixScale(Normalization):
    '''Normalize by division through a defined *factor*.'''

    factor = Float.T()

    def __call__(self, chunk):
        chunk[:] /= self.factor


class NormalizeNthRoot(Normalization):
    '''Normalize traces by their Nth root.

    Note: polarities get lost.'''
    nth_root = Int.T(default=4)

    def __call__(self, chunk):
        chunk -= num.nanmean(chunk)
        chunk[:] = num.power(num.abs(chunk), 1./self.nth_root)
        chunk += num.nanmean(chunk)


class NormalizeChannelMax(Normalization):
    '''Normalize channel wise by division through the maximum of each channel.
    '''
    def __call__(self, chunk):
        chunk /= ((num.nanmax(num.abs(chunk), axis=1)[:, num.newaxis]) + EPSILON )


class NormalizeStd(Normalization):
    '''Normalizes by dividing through the standard deviation'''
    def __call__(self, chunk):
        # save and subtract trace levels
        trace_levels = num.nanmean(chunk)
        chunk -= trace_levels

        # normalize
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


class Noise(ChunkOperation):
    level = Float.T(default=1., optional=True)

    def __call__(self, chunk):
        pass


class WhiteNoise(Noise):
    '''Add white noise to data.'''

    def __call__(self, chunk):
        chunk += num.random.random(chunk.shape) * self.level


class Imputation(ChunkOperation):
    pass


class ImputationZero(Imputation):
    def __call__(self, chunk):
        return 0.

class ImputationMin(Imputation):

    def __call__(self, chunk):
        return num.nanmin(chunk) + EPSILON


class ImputationMean(Imputation):

    def __call__(self, chunk):
        return num.nanmean(chunk) + EPSILON


class ImputationMedian(Imputation):

    def __call__(self, chunk):
        return num.nanmedian(chunk) + EPSILON


class DataGeneratorBase(Object):
    '''This is the base class for all generators.

    This class to dump and load data to/from all subclasses into
    TFRecordDatasets.
    '''
    fn_tfrecord = String.T(optional=True)
    noise = Noise.T(optional=True, help='Add noise to feature')

    station_dropout_rate = Float.T(default=0.,
        help='Rate by which to mask all channels of station')

    station_dropout_distribution = Bool.T(default=True,
        help='If *true*, station dropout will be drawn from a uniform '
        'distribution limited by this station_dropout.')

    nmax = Int.T(optional=True)
    labeled = Bool.T(default=True)
    blacklist = List.T(optional=True, help='List of indices to ignore.')

    random_seed = Int.T(default=0)

    def __init__(self, *args, **kwargs):
        self.config = kwargs.pop('config', None)
        super().__init__(**kwargs)
        self.blacklist = set() if not self.blacklist else set(self.blacklist)
        self.n_classes = self.config.n_classes
        self.evolution = 0

    def normalize_label(self, label):
        if self.labeled:
            return self.config.normalize_label(label)
        return label

    def set_config(self, pinky_config):
        self.config = pinky_config
        self.setup()

    def setup(self):
        ...

    def reset(self):
        self.evolution = 0

    @property
    def tensor_shape(self):
        return self.config.tensor_shape

    @property
    def n_samples(self):
        return self.config._n_samples

    @n_samples.setter
    def n_samples(self, v):
        self.config._n_samples = v

    @property
    @lru_cache(maxsize=1)
    def nsl_to_indices(self):
        ''' Returns a dictionary which maps nsl codes to indexing arrays.'''
        indices = OrderedDict()
        for nslc, index in self.nslc_to_index.items():
            key = nslc[:3]
            _v = indices.get(key, [])
            _v.append(index)
            indices[key] = _v

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
        for nslc in self.config.channels:
            if not util.match_nslc(self.config.blacklist, nslc):
                d[nslc] = idx
                idx += 1
        return d

    def reject_blacklisted(self, tr):
        '''returns `False` if nslc codes of `tr` match any of the blacklisting
        patters. Otherwise returns `True`'''
        return not util.match_nslc(self.config.blacklist, tr.nslc_id)

    def filter_iter(self, iterator):
        '''Apply *blacklist*ing by example indices

        :param iterator: producing iterator
        '''
        for i, item in enumerate(iterator):
            if i not in self.blacklist:
                yield i, item

    @property
    def generate_output_types(self):
        '''Return data types of features and labels'''
        return tf.float32, tf.float32

    def unpack_examples(self, record_iterator):
        '''Parse examples stored in TFRecordData to `tf.train.Example`'''
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            chunk = example.features.feature['data'].bytes_list.value[0]
            label = example.features.feature['label'].bytes_list.value[0]

            chunk = num.fromstring(chunk, dtype=num.float32)
            chunk = chunk.reshape((self.config.n_channels, -1))

            label = num.fromstring(label, dtype=num.float32)
            yield chunk, label

    @property
    def tstart_data(self):
        return None

    def iter_chunked(self, tinc):
        # if data has been written to tf records:
        return self.iter_examples_and_labels()

    def iter_examples_and_labels(self):
        '''Subclass this method!

        Yields: feature, label

        Chunks that are all NAN will be skipped.
        '''
        record_iterator = tf.python_io.tf_record_iterator(
            path=self.fn_tfrecord)

        for chunk, label in self.unpack_examples(record_iterator):
            if all_NAN(chunk):
                logger.debug('all NAN. skipping...')
                continue

            yield chunk, label

    def generate_chunked(self, tinc=1):
        '''Takes the output of `iter_examples_and_labels` and applies post
        processing (see: `process_chunk`).
        '''
        for i, (chunk, label) in self.filter_iter(self.iter_chunked(tinc)):
            yield self.process_chunk(chunk), self.normalize_label(label)

    def generate(self, return_gaps=False):
        '''Takes the output of `iter_examples_and_labels` and applies post
        processing (see: `process_chunk`).
        '''
        self.evolution += 1
        num.random.seed(self.random_seed + self.evolution)
        for i, (chunk, label) in self.filter_iter(
                self.iter_examples_and_labels()):
            yield self.process_chunk(chunk, return_gaps=return_gaps), self.normalize_label(label)

    def extract_labels(self):
        '''Overwrite this method!'''
        if not self.labeled:
            return UNLABELED

    def iter_labels(self):
        '''Iterate through labels.'''
        for i, (_, label) in self.filter_iter(
                self.iter_examples_and_labels()):
            yield label

    @property
    def text_labels(self):
        '''Returns a list of strings to identify the labels.

        Overwrite this method for more meaningfull identifiers.'''
        return ['%i' % (i) for i, d in
                self.filter_iter(self.iter_examples_and_labels())]

    def gaps(self):
        '''Returns a list containing the gaps of each example'''
        gaps = []
        for (_, gap), _ in self.generate(return_gaps=True):
            gaps.append(gap)

        return gaps

    def snrs(self, split_factor):
        snrs = []
        for chunk, _ in self.generate():
            snrs.append(snr(chunk, split_factor))
        return snrs

    @property
    def output_shapes(self):
        return (self.config.output_shapes)

    def get_dataset(self):
        return tf.data.Dataset.from_generator(
            self.generate,
            self.generate_output_types,
            output_shapes=self.output_shapes)

    def get_chunked_dataset(self, tinc=1.):
        gen = partial(self.generate_chunked, tinc=tinc)
        return tf.data.Dataset.from_generator(
            gen,
            self.generate_output_types,
            output_shapes=self.output_shapes)

    def get_raw_data_chunk(self, shape):
        '''Return an array of size (Nchannels x Nsamples_max) filled with
        NANs.'''
        empty_array = num.empty(shape, dtype=num.float32)
        empty_array.fill(num.nan)
        return empty_array

    def pack_examples(self):
        '''Serialize Examples to strings.'''
        for ydata, label in self.iter_examples_and_labels():
            yield tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'data': _BytesFeature(ydata.tobytes()),
                        'label': _BytesFeature(num.array(
                            label, dtype=num.float32).tobytes()),
                    }))

    def mask(self, chunk, rate):
        '''For data augmentation: Mask traces in chunks with NaNs.
        NaNs will be filled by the imputation method provided by the config
        file.

        :param rate: probability with which traces are NaN-ed
        '''
        # print(rate)
        indices = self.nsl_indices
        a = num.random.random(len(indices))
        i = num.where(a < rate)[0]
        for ii in i:
            chunk[indices[ii], :] = num.nan

    def random_trim(self, chunk, margin):
        '''For data augmentation: Randomly trim examples in time domain with
        *margin* seconds.'''
        sample_margin = int(margin / self.config.effective_deltat)
        nstart = num.random.randint(low=0, high=sample_margin)

        _, n_samples = self.config.tensor_shape
        nstop = nstart + n_samples
        chunk[:, :nstart] = 0.
        chunk[:, nstop:] = 0.

    def process_chunk(self, chunk, return_gaps=False):
        '''Performs preprocessing of data chunks.'''

        if self.config.t_translation_max:
            self.random_trim(chunk, self.config.t_translation_max)

        # add noise
        if self.noise:
            self.noise(chunk)

        # apply normalization
        self.config.normalization(chunk)

        # apply station dropout
        if self.station_dropout_rate:
            if self.station_dropout_distribution:
                self.mask(chunk, num.random.uniform(
                    high=self.station_dropout_rate))
            else:
                self.mask(chunk, self.station_dropout_rate)

        # fill gaps
        if self.config.imputation:
            gaps = num.isnan(chunk)
            chunk[gaps] = self.config.imputation(chunk)

        if not return_gaps:
            return chunk
        else:
            return chunk, gaps

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
    in_generator = DataGeneratorBase.T(help='The generator to be compressed')

    def setup(self):
        self.nsl_to_indices_orig = copy.deepcopy(
                self.in_generator.nsl_to_indices)

        self._channels = [k + ('STACK', ) for k in
                self.in_generator.nsl_to_indices.keys()]
        self.n_channels = len(self._channels)

    @property
    def nslc_to_index(self):
        d = OrderedDict()
        for idx, nsl in enumerate(self._channels):
            d[nsl[:3]] = idx
        return d

    @property
    def tensor_shape(self):
        return (len(self.nslc_to_index), self.config.tensor_shape[1])

    @property
    def output_shapes(self):
        '''Return a tuple containing the shape of feature arrays and number of
        labels.
        '''
        return (self.tensor_shape, self.config.n_classes)

    def iter_chunked(self, tinc):
        d = self.nslc_to_index
        for feature, label in self.in_generator.iter_chunked(tinc=tinc):
            chunk = self.get_raw_data_chunk(shape=self.tensor_shape)
            for nsl, indices in self.nsl_to_indices_orig.items():
                chunk[d[nsl]] = num.sum(num.abs(feature[indices, :]), axis=0)

            if all_NAN(chunk):
                logger.debug('all NAN. skipping...')
                continue

            yield chunk, label

    def iter_examples_and_labels(self):
        d = self.nslc_to_index
        for feature, label in self.in_generator.iter_examples_and_labels():
            chunk = self.get_raw_data_chunk(shape=self.tensor_shape)
            for nsl, indices in self.nsl_to_indices_orig.items():
                chunk[d[nsl]] = num.sum(num.abs(feature[indices, :]), axis=0)

            if all_NAN(chunk):
                logger.debug('all NAN. skipping...')
                continue

            yield chunk, label

    @classmethod
    def from_generator(cls, generator):
        x = cls(in_generator=generator, config=generator.config)
        x.station_dropout_rate = generator.station_dropout_rate
        x.setup()
        return x

    def iter_labels(self):
        return self.in_generator.iter_labels()
    
    @property
    def tstart_data(self):
        return self.in_generator.tstart_data

class DataGenerator(DataGeneratorBase):

    def preprocess(self, tr):
        '''Trace preprocessing

        :param tr: pyrocko.tr.Trace object

        Demean, type casting to float32, filtering, adjust sampling rates.
        '''
        tr.ydata = tr.ydata.astype(num.float32)
        tr.ydata -= num.mean(tr.ydata)

        filter_function = self.get_filter_function()
        filter_function(tr)

        dt_want = self.config.deltat_want
        if dt_want is not None:
            if tr.deltat - dt_want > EPSILON:
                tr.resample(dt_want)
            elif tr.deltat - dt_want < -EPSILON:
                tr.downsample_to(dt_want)

    @lru_cache(maxsize=1)
    def get_filter_function(self):
        '''Setup and return a function that takes applies high- and lowpass
        filters to :py:class:`pyrocko.trace.Trace` instances.
        '''
        functions = []
        if self.config.highpass is not None:
            functions.append(lambda tr: tr.highpass(
                corner=self.config.highpass,
                order=self.config.highpass_order))
        if self.config.lowpass is not None:
            functions.append(lambda tr: tr.lowpass(
                corner=self.config.lowpass,
                order=self.config.lowpass_order))

        def fn(t):
            for f in functions:
                f(t)
            if self.config.absolute:
                t.ydata = num.abs(t.ydata)

        return fn

    def extract_labels(self, source):
        if not self.labeled:
            return UNLABELED

        n, e = orthodrome.latlon_to_ne(
            self.config.reference_target.lat, self.config.reference_target.lon,
            source.lat, source.lon)

        return (n, e, source.depth)

    def fit_data_into_chunk(self, traces, chunk, indices=None, tref=0):
        '''Fit all `traces` into a 2 demensional numpy array.

        :param traces: list of pyrocko.trace.Trace instances
        :param indices: list of indices where the traces in `traces` are
            supposed to be filled into the array. If this parameter is None
            assumes that the order of traces will remain the same accross
            iterations.
        :param tref: absolute time where to chop the data chunk. Typically first
            p phase onset'''
        indices = indices or range(len(traces))
        tref = tref - self.config.effective_tpad
        for i, tr in zip(indices, traces):
            data_len = len(tr.ydata)
            istart_trace = int((tr.tmin - tref) / tr.deltat)
            istart_array = max(istart_trace, 0)
            istart_trace = max(-istart_trace, 0)
            istop_array = istart_array + (data_len - 2* istart_trace)

            ydata = tr.ydata[
                istart_trace: min(data_len, self.n_samples-istart_array) \
                    + istart_trace]
            chunk[i, istart_array: istart_array+ydata.shape[0]] = ydata


class PileData(DataGenerator):
    '''Data generator for locally saved data.'''
    fn_stations = String.T()
    data_paths = List.T(String.T())
    data_format = String.T(default='detect')
    fn_markers = String.T()
    fn_events = String.T(optional=True)
    sort_markers = Bool.T(default=False,
            help= 'Sorting markers speeds up data io. Shuffled markers \
            improve generalization')
    align_phase = String.T(default='P')

    tstart = String.T(optional=True)
    tstop = String.T(optional=True)

    def setup(self):
        self.data_pile = pile.make_pile(
            self.data_paths, fileformat=self.data_format)

        if self.data_pile.is_empty():
            sys.exit('Data pile is empty!')

        self.deltat_want = self.config.deltat_want or \
                min(self.data_pile.deltats.keys())

        self.n_samples = int(
                (self.config.sample_length + self.config.tpad) / self.deltat_want)

        logger.debug('loading marker file %s' % self.fn_markers)

        # loads just plain markers:
        markers = marker.load_markers(self.fn_markers)

        if self.fn_events:
            markers.extend(
                    [marker.EventMarker(e) for e in
                load_events(self.fn_events)])

        if self.sort_markers:
            logger.info('sorting markers!')
            markers.sort(key=lambda x: x.tmin)
        marker.associate_phases_to_events(markers)

        markers_by_nsl = {}
        for m in markers:
            if not m.match_nsl(self.config.reference_target.codes[:3]):
                continue

            if m.get_phasename().upper() != self.align_phase:
                continue

            markers_by_nsl.setdefault(m.one_nslc()[:3], []).append(m)

        assert(len(markers_by_nsl) == 1)

        # filter markers that do not have an event assigned:
        self.markers = list(markers_by_nsl.values())[0]

        if not self.labeled:
            dummy_event = Event(lat=0., lon=0., depth=0.)
            for m in self.markers:
                if not m.get_event():
                    m.set_event(dummy_event)

        self.markers = [m for m in self.markers if m.get_event() is not None]

        if not len(self.markers):
            raise Exception('No markers left in dataset')

        self.config.channels = list(self.data_pile.nslc_ids.keys())
        self.config.channels.sort()

    def check_inputs(self):
        if len(self.data_pile.deltats()) > 1:
            logger.warn(
                'Different sampling rates in dataset. Preprocessing slow')

    def extract_labels(self, marker):
        if not self.labeled:
            return UNLABELED

        source = marker.get_event()
        n, e = orthodrome.latlon_to_ne(
            self.config.reference_target.lat, self.config.reference_target.lon,
            source.lat, source.lon)
        return (n, e, source.depth)

    @property
    def tstart_data(self):
        '''Returns start point of data returned by generator.'''
        return util.stt(self.tstart) if self.tstart else self.data_pile.tmin

    def iter_chunked(self, tinc):
        tr_len = self.n_samples * self.deltat_want
        nslc_to_index = self.nslc_to_index

        tpad = self.config.effective_tpad

        tstart = util.stt(self.tstart) if self.tstart else None
        tstop = util.stt(self.tstop) if self.tstop else None

        logger.debug('START')
        for trs in self.data_pile.chopper(
                tinc=tinc, tmin=tstart, tmax=tstop, tpad=tpad,
                keep_current_files_open=True, want_incomplete=False,
                trace_selector=self.reject_blacklisted):

            chunk = self.get_raw_data_chunk(self.tensor_shape)

            if not trs:
                yield chunk, UNLABELED
                continue

            for tr in trs:
                self.preprocess(tr)

            indices = [nslc_to_index[tr.nslc_id] for tr in trs]
            self.fit_data_into_chunk(trs, chunk=chunk, indices=indices,
                    tref=trs[0].tmin)

            if all_NAN(chunk):
                logger.debug('all NAN. skipping...')
                continue

            yield chunk, UNLABELED

    def iter_labels(self):
        for m in self.markers:
            yield self.extract_labels(m)

    def iter_examples_and_labels(self):
        tr_len = self.n_samples * self.deltat_want
        nslc_to_index = self.nslc_to_index

        tpad = self.config.effective_tpad

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
                chunk = self.get_raw_data_chunk(self.tensor_shape)
                self.fit_data_into_chunk(trs, chunk=chunk, indices=indices, tref=m.tmin)

                if all_NAN(chunk):
                    logger.debug('all NAN. skipping...')
                    continue

                label = self.extract_labels(m)
                yield chunk, label


class SeismosizerData(DataGenerator):
    fn_sources = String.T(
            help='filename containing pyrocko.gf.seismosizer.Source instances')
    fn_targets = String.T(
            help='filename containing pyrocko.gf.seismosizer.Target instances',
            optional=True)
    fn_stations = String.T(
            help='filename containing pyrocko.model.Station instances. Will be\
                converted to Target instances',
            optional=True)

    store_id = String.T(optional=True)
    center_sources = Bool.T(
            default=False,
            help='Transform the center of sources to the center of stations')

    engine = LocalEngine.T()
    onset_phase = String.T(default='first(p|P)')

    def setup(self):
        self.sources = guts.load(filename=self.fn_sources)
        self.targets = []

        if self.fn_targets:
            self.targets.extend(guts.load(filename=self.fn_targets))

        if self.fn_stations:
            stats = load_stations(self.fn_stations)
            self.targets.extend(self.cast_stations_to_targets(stats))

        if self.store_id:
            for t in self.targets:
                t.store_id = self.store_id

        if self.center_sources:
            self.move_sources_to_station_center()

        self.config.channels = [t.codes for t in self.targets]
        store_ids = [t.store_id for t in self.targets]
        store_id = set(store_ids)
        assert len(store_id) == 1, 'More than one store used. Not \
                implemented yet'

        self.store = self.engine.get_store(store_id.pop())

        self.sources = filter_oob(self.sources, self.targets, self.store.config)

        dt = self.config.deltat_want or self.store.config.deltat
        self.n_samples = int((self.config.sample_length + self.config.tpad) / dt)

    def move_sources_to_station_center(self):
        '''Transform the center of sources to the center of stations.'''
        lat, lon = orthodrome.geographic_midpoint_locations(self.targets)
        for s in self.sources:
            s.lat = lat
            s.lon = lon

    def cast_stations_to_targets(self, stations):
        targets = []
        channels = 'ENZ'
        for s in stations:
            targets.extend(
                [Target(codes=(s.network, s.station, s.location, c),
                    lat=s.lat, lon=s.lon, elevation=s.elevation,) for c in
                    channels])

        return targets

    def extract_labels(self, source):
        if not self.labeled:
            return UNLABELED
        return (source.north_shift, source.east_shift, source.depth)

    def iter_examples_and_labels(self):
        ensure_list(self.sources)
        ensure_list(self.targets)

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
            chunk = self.get_raw_data_chunk(self.tensor_shape)

            self.fit_data_into_chunk(traces, chunk=chunk, tref=tref+source.time)

            label = self.extract_labels(source)

            yield chunk, label


name_to_class = {
        'NormalizeMax': NormalizeMax,
        'NormalizeLog': NormalizeLog,
        'NormalizeNthRoot': NormalizeNthRoot,
        'NormalizeChannelMax': NormalizeChannelMax,
        'NormalizeChannelStd': NormalizeChannelStd,
        'NormalizeStd': NormalizeStd,
        'ImputationMin': ImputationMin,
        'ImputationMean': ImputationMean,
        'ImputationMedian': ImputationMedian,
        'ImputationZero': ImputationZero}

