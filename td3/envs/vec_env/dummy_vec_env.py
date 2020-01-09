import numpy as np
from rllab.core.serializable import Serializable
from rllab.envs.base import Env, Step

class DummyVecEnv(Env, Serializable):
    def __init__(self, vec_env):
        Serializable.quick_init(self, locals())
        self._vec_env = vec_env

    def reset(self):
        results = [env.reset() for env in self._vec_env]
        return np.array(results)

    @property
    def action_space(self):
        return self._vec_env[0].action_space

    @property
    def observation_space(self):
        return self._vec_env[0].observation_space

    def step(self, action):
        results = []
        for (a, env) in zip(action, self._vec_env):
            next_obs, reward, done, info = env.step(a)
            results.append(Step(next_obs, reward, done, **info))
        return results

    def render(self, *args, **kwargs):
        return self._vec_env[0].render(*args, **kwargs)

    def log_diagnostics(self, paths, *args, **kwargs):
        [env.log_diagnostics(paths, *args, **kwargs) for env in self._vec_env]

    def get_param_values(self):
        return [env.get_param_values() for env in self._vec_env]

    def set_param_values(self, params):
        [env.set_param_values(param) for env, param in zip(self._vec_env, params)]

    @property
    def num_envs(self):
        return len(self._vec_env)

    @property
    def envs(self):
        return self._vec_env

    def __str__(self):
        return "Dummy %d environments: %s" %(self.num_envs, self._vec_env[0])


dummy = DummyVecEnv