import tensorflow as tf


def _Int64Feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _FloatFeature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _BytesFeature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def variable_summaries(var, name_scope=None):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    scope = name_scope or 'name_scope'
    with tf.name_scope(scope):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
