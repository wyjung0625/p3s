import tensorflow as tf
from rllab import config


def get_default_session():
    return tf.get_default_session() or create_session()


def create_session(**kwargs):
    """ Create new tensorflow session with given configuration. """
    if "config" not in kwargs:
        kwargs["config"] = get_configuration()
    return tf.InteractiveSession(**kwargs)


def get_configuration():
    """ Returns personal tensorflow configuration. """
    if config.USE_GPU:
        raise NotImplementedError

    config_args = dict()
    return tf.ConfigProto(**config_args)





"""
=============================================================
tf_util from baselines of OPEN AI
=============================================================
"""

_PLACEHOLDER_CACHE = {}  # name -> (placeholder, dtype, shape)

def get_placeholder(name, dtype, shape):
    if name in _PLACEHOLDER_CACHE:
        out, dtype1, shape1 = _PLACEHOLDER_CACHE[name]
        assert dtype1 == dtype and shape1 == shape
        return out
    else:
        out = tf.placeholder(dtype=dtype, shape=shape, name=name)
        _PLACEHOLDER_CACHE[name] = (out, dtype, shape)
        return out

def get_placeholder_name():
    for name in _PLACEHOLDER_CACHE:
        print(name)
    return _PLACEHOLDER_CACHE