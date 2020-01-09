import numpy as np

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv
from rllab.spaces.box import Box
from rllab.misc.overrides import overrides
from rllab.envs.base import Step


class DelayedEnv(ProxyEnv, Serializable):
    def __init__(
            self,
            env,
            reward_freq=5,
    ):
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        self._reward_freq = reward_freq
        self._delayed_reward = 0.
        self._delayed_step = 0

    def reset(self):
        return self._wrapped_env.reset()

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d["_reward_freq"] = self._reward_freq
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self._reward_freq = d["_reward_freq"]

    @property
    @overrides
    def action_space(self):
        if isinstance(self._wrapped_env.action_space, Box):
            ub = np.ones(self._wrapped_env.action_space.shape)
            return spaces.Box(-1 * ub, ub)
        return self._wrapped_env.action_space

    @overrides
    def step(self, action):
        if isinstance(self._wrapped_env.action_space, Box):
            # rescale the action
            lb, ub = self._wrapped_env.action_space.bounds
            scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
            scaled_action = np.clip(scaled_action, lb, ub)
        else:
            scaled_action = action
        wrapped_step = self._wrapped_env.step(scaled_action)

        next_obs, reward, done, info = wrapped_step

        self._delayed_reward += reward
        self._delayed_step += 1

        # print("delayed step : %d" % self._delayed_step)
        # print("done : ", done)

        if done or self._delayed_step == self._reward_freq:
            wrapped_reward = self._delayed_reward
            self._delayed_reward = 0
            self._delayed_step = 0
        else:
            wrapped_reward = 0

        # print("delayed reward : %.2f" % self._delayed_reward)
        # print("wrapped reward : %.2f" % wrapped_reward)
        return Step(next_obs, wrapped_reward, done, **info)

    def __str__(self):
        return "Reward frequency: %d" % self._reward_freq

    # def log_diagnostics(self, paths):
    #     print "Obs mean:", self._obs_mean
    #     print "Obs std:", np.sqrt(self._obs_var)
    #     print "Reward mean:", self._reward_mean
    #     print "Reward std:", np.sqrt(self._reward_var)

delay = DelayedEnv
