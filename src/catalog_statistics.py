from pyrocko.gui.marker import PhaseMarker, EventMarker, associate_phases_to_events
from pyrocko.model.event import load_events
from pyrocko import util

import numpy as num

debug = True


def onset_stats(fn_markers, fn_events, fn_events_all):
    markers = PhaseMarker.load_markers(fn_markers)

    markers_all = [EventMarker(e) for e in load_events(fn_events_all)]
    associate_phases_to_events(markers_all)
    tt_by_nsl = get_tt_by_nsl(markers_all)

    if debug:
        plot_onsets(tt_by_nsl)


def power(tr):
    return num.sqrt(num.sum(tr.get_ydata()**2))/(tr.tmax - tr.tmin)


def normalize(tr):
    tr.ydata = tr.ydata / power(tr)


def get_tt_by_nsl(markers):
    ''' Returns a dictionary with nsl as keys and travel times as value list'''
    onsets = {}
    for m in markers:
        if not isinstance(m, PhaseMarker):
            continue
        if not m.get_phasename().upper() != 'P':
            continue

        e = m.get_event()
        if not e:
            continue

        tt = m.tmin - e.time

        # remove some outliers:
        if 0 < tt < 14:
            append_to_dict(onsets, m.one_nslc()[:3], tt)

    return onsets


def get_average_onsets(markers):
    onsets = {}
    for m in markers:
        if not isinstance(m, PhaseMarker):
            continue
        if not m.get_phasename().upper() != 'P':
            continue
        e = m.get_event()
        if e is not None:
            append_to_dict(
                onsets, m.one_nslc()[:3], m.tmin - e.time)

    for k, v in onsets.items():
        onsets[k] = num.mean(v)

    return onsets


def get_average_scaling(trs_dict, reference_nsl_pattern):

    scalings = {}
    for event, trs in trs_dict.items():
        refs = [tr for tr in trs if util.match_nslc(
            reference_nsl_pattern, tr.nslc_id)]
        for ref in refs:
            for tr in trs:
                if tr.channel == ref.channel:
                    append_to_dict(
                        scalings, tr.nslc_id, power(tr)/power(ref))

    for k, v in scalings.items():
        scalings[k] = num.mean(v)

    return scalings


def two_sided_percentile(vals, percentile=95.):
    p = percentile
    vals = num.array(vals)
    return num.percentile(vals, p), -1. * num.percentile(-1.*vals, p)


def plot_onsets(onsets):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    vals = list(onsets.values())
    for i, v in enumerate(vals):
        ax = fig.add_subplot(4, 1 + len(onsets)//4, i+1)
        v = num.array(v)
        ax.hist(v, bins=31)
        ax.axvspan(*two_sided_percentile(v), alpha=0.2)
    plt.show()

if __name__ == '__main__':
    fn_markers = '/home/marius/josef_dd/markers_with_polarities.pf'
    fn_events =  '/home/marius/josef_dd/events_from_sebastian_subset.pf'
    onset_stats(fn_markers,fn_events, fn_events)
