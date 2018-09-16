import os
import shutil
import logging

logger = logging.getLogger()


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def first_element(x):
    if len(x) <= 1:
        return x[0]
    elif len(x) > 1:
        raise Exception('%s has more than one item' % x)


def filter_oob(sources, targets, config):
    '''Filter sources that will be out of bounds of GF database.'''
    nsources, ntargets = len(sources), len(targets)
    slats, slons = num.empty(nsources), num.empty(nsources)
    sdepth = num.empty(nsources)
    tlats, tlons = num.empty(ntargets), num.empty(ntargets)

    for i_s, s in enumerate(sources):
        slats[i_s], slons[i_s], sdepth[i_s] = *s.effective_latlon, s.depth

    for i_t, t in enumerate(targets):
        tlats[i_t], tlons[i_t] = t.effective_latlon

    dists = num.empty((ntargets, nsources))
    for i in range(ntargets):
        dists[i] = orthodrome.distance_accurate50m_numpy(
                slats, slons, tlats[i], tlons[i])

    i_dist = num.logical_or(
            dists > config.distance_max, dists < config.distance_min)

    i_depth = num.logical_or(
            sdepth > config.source_depth_max, sdepth < config.source_depth_min)

    i_filter = num.where(num.any(num.logical_or(i_dist, i_depth), axis=0))[0]
    i_filter.sort()

    logger.debug('Removing %i sources which would be out of bounds' %
            len(i_filter))

    for i in i_filter[::-1]:
        del sources[i]


def delete_if_exists(dir_or_file):
    '''Deletes `dir_or_file` if exists'''
    if os.path.exists(dir_or_file):
        if os.path.isfile(dir_or_file):
            logger.debug('deleting file: %s' % dir_or_file)
            os.remove(dir_or_file)
        else:
            logger.debug('deleting directory: %s' % dir_or_file)
            shutil.rmtree(dir_or_file)


def nsl(tr):
    return tr.nslc_id[:3]


def append_to_dict(d, k, v):
    _m = d.get(k, [])
    _m.append(v)
    d[k] = _m

