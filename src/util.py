import os
import shutil
import logging

logger = logging.getLogger()


def delete_if_exists(dirname):
    if os.path.exists(dirname):
        logger.info('deleting directory: %s' % dirname)
        shutil.rmtree(dirname)


def nsl(tr):
    return tr.nslc_id[:3]


def append_to_dict(d, k, v):
    _m = d.get(k, [])
    _m.append(v)
    d[k] = _m

