from contextlib import contextmanager
import numpy as np
import tensorflow as tf

from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.core.serializable import Serializable

from td3.policies import NNPolicy
from td3.misc import tf_utils
from td3.misc.mlp import mlp

EPS = 1e-6


class DeterministicPolicy(NNPolicy, Serializable):
    def __init__(self, env_spec, hidden_layer_sizes=(100, 100), reg=1e-3,
                 squash=True, name='deterministic_policy', observation_ph=None, noise_scale=0.1):
        """
        Args:
            env_spec (`rllab.EnvSpec`): Specification of the environment
                to create the policy for.
            hidden_layer_sizes (`list` of `int`): Sizes for the Multilayer
                perceptron hidden layers.
            reg (`float`): Regularization coeffiecient for the Gaussian parameters.
            squash (`bool`): If True, squash the Gaussian the gmm action samples
               between -1 and 1 with tanh.
            reparameterize ('bool'): If True, gradients will flow directly through
                the action samples.
        """
        Serializable.quick_init(self, locals())


        self._Da = env_spec.action_space.flat_dim
        self._Ds = env_spec.observation_space.flat_dim
        self._hidden_layers = list(hidden_layer_sizes) + [self._Da]
        self._squash = squash
        self._reg = reg
        self._noise_scale = noise_scale
        self._max_actions = int(env_spec.action_space.high[0])
        self._is_deterministic = False

        self._observations_ph = observation_ph if observation_ph is not None else tf_utils.get_placeholder(name='observation', dtype=tf.float32, shape=[None, self._Ds])

        self.name = name
        self.build()

        self._scope_name = (
            tf.get_variable_scope().name + "/" + name
        ).lstrip("/")

        super(NNPolicy, self).__init__(env_spec)

    def actions_for(self, observations,
                    name=None, reuse=tf.AUTO_REUSE):
        name = name or self.name

        with tf.variable_scope(name, reuse=reuse):
            if self._squash:
                output_nonlinearity = tf.nn.tanh
            else:
                output_nonlinearity = None

            self._actions = mlp(
                inputs=(observations,),
                layer_sizes=self._hidden_layers,
                output_nonlinearity=output_nonlinearity,
            )

        return self._actions

    def dist(self, other):
        return tf.reduce_mean(0.5 * tf.square(other._actions - self._actions), axis=-1)

    def build(self):
        with tf.variable_scope(self.name):
            if self._squash:
                output_nonlinearity = tf.nn.tanh
            else:
                output_nonlinearity = None

            self._actions = mlp(
                inputs=(self._observations_ph,),
                layer_sizes=self._hidden_layers,
                output_nonlinearity=output_nonlinearity,
            )

    @overrides
    def get_actions(self, observations):
        feed_dict = {self._observations_ph: observations}
        actions = np.array(tf.get_default_session().run(self._actions, feed_dict))

        if self._is_deterministic:
            return actions
        else:
            noise = self._noise_scale * np.random.randn(actions.shape[0], actions.shape[1])
            noisy_actions = np.clip(actions + noise, -self._max_actions, self._max_actions)

            return noisy_actions

    @contextmanager
    def deterministic(self, set_deterministic=True, latent=None):
        """Context manager for changing the determinism of the policy.

        See `self.get_action` for further information about the effect of
        self._is_deterministic.

        Args:
            set_deterministic (`bool`): Value to set the self._is_deterministic
                to during the context. The value will be reset back to the
                previous value when the context exits.
            latent (`Number`): Value to set the latent variable to over the
                deterministic context.
        """
        was_deterministic = self._is_deterministic

        self._is_deterministic = set_deterministic

        yield

        self._is_deterministic = was_deterministic

    @property
    def action_t(self):
        return self._actions


