import tensorflow as tf

from rllab.core.serializable import Serializable

from td3.misc.mlp import MLPFunction
from td3.misc import tf_utils

class NNVFunction(MLPFunction):

    def __init__(self, env_spec, hidden_layer_sizes=(100, 100), name='vf', observation_ph=None):
        Serializable.quick_init(self, locals())

        self._Do = env_spec.observation_space.flat_dim
        self._obs_pl = observation_ph if observation_ph is not None else tf_utils.get_placeholder(name='observation', dtype=tf.float32, shape=[None, self._Do])
        self.name = name

        super(NNVFunction, self).__init__(
            name, (self._obs_pl,), hidden_layer_sizes)


class NNQFunction(MLPFunction):
    def __init__(self, env_spec, hidden_layer_sizes=(100, 100), name='qf', observation_ph=None, action_ph=None):
        Serializable.quick_init(self, locals())

        self._Da = env_spec.action_space.flat_dim
        self._Do = env_spec.observation_space.flat_dim

        self._obs_pl = observation_ph if observation_ph is not None else tf_utils.get_placeholder(name='observation', dtype=tf.float32, shape=[None, self._Do])
        self._action_pl = action_ph if action_ph is not None else tf_utils.get_placeholder(name='actions', dtype=tf.float32, shape=[None, self._Da])

        self.name = name

        super(NNQFunction, self).__init__(
            name, (self._obs_pl, self._action_pl), hidden_layer_sizes)
