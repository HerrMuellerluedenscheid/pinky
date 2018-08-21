import os
import shutil
import logging

logger = logging.getLogger()


def first_element(x):
    if len(x) <= 1:
        return x[0]
    elif len(x) > 1:
        raise Exception('%s has more than one item' % x)


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

