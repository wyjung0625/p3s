import numpy as np
import time
from collections import deque

from rllab.misc import logger
from rllab.misc.overrides import overrides


def rollout(env, policy, path_length, render=False, speedup=None):
    Da = env.action_space.flat_dim
    Do = env.observation_space.flat_dim

    observation = env.reset()
    policy.reset()

    observations = np.zeros((path_length + 1, Do))
    actions = np.zeros((path_length, Da))
    terminals = np.zeros((path_length, ))
    rewards = np.zeros((path_length, ))
    agent_infos = []
    env_infos = []

    t = 0
    for t in range(path_length):

        action, agent_info = policy.get_action(observation)
        next_obs, reward, terminal, env_info = env.step(action)

        agent_infos.append(agent_info)
        env_infos.append(env_info)

        actions[t] = action
        terminals[t] = terminal
        rewards[t] = reward
        observations[t] = observation

        observation = next_obs

        if render:
            env.render()
            time_step = 0.05
            time.sleep(time_step / speedup)

        if terminal:
            break

    observations[t + 1] = observation

    path = {
        'observations': observations[:t + 1],
        'actions': actions[:t + 1],
        'rewards': rewards[:t + 1],
        'terminals': terminals[:t + 1],
        'next_observations': observations[1:t + 2],
        'agent_infos': agent_infos,
        'env_infos': env_infos
    }

    return path


def rollouts(env, policy, path_length, n_paths):
    paths = list()
    for i in range(n_paths):
        paths.append(rollout(env, policy, path_length))

    return paths


class Sampler(object):
    def __init__(self, max_path_length, min_pool_size, batch_size):
        self._max_path_length = max_path_length
        self._min_pool_size = min_pool_size
        self._batch_size = batch_size

        self.env = None
        self.policy = None
        self.pool = None

    def initialize(self, env, policy, pool):
        self.env = env
        self.policy = policy
        self.pool = pool

    def set_policy(self, policy):
        self.policy = policy

    def sample(self):
        raise NotImplementedError

    def batch_ready(self):
        enough_samples = self.pool.size >= self._min_pool_size
        return enough_samples

    def random_batch(self):
        return self.pool.random_batch(self._batch_size)

    def terminate(self):
        self.env.terminate()

    def log_diagnostics(self):
        logger.record_tabular('pool-size', self.pool.size)


class SimpleSampler(Sampler):
    def __init__(self, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)

        self._path_length = 0
        self._path_return = 0
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0

        self._max_episodes = 10
        self._avg_return = deque([], maxlen=self._max_episodes)

    def sample(self):
        if self._current_observation is None:
            self._current_observation = self.env.reset()

        action, _ = self.policy.get_action(self._current_observation)
        next_observation, reward, terminal, info = self.env.step(action)
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1


        self.pool.add_sample(
            observation=self._current_observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation)

        if terminal or self._path_length >= self._max_path_length:
            self.policy.reset()
            self._current_observation = self.env.reset()
            self._path_length = 0
            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return
            self._avg_return.extend([self._path_return])

            self._path_return = 0
            self._n_episodes += 1

        else:
            self._current_observation = next_observation

    def log_diagnostics(self):
        super(SimpleSampler, self).log_diagnostics()
        # logger.record_tabular('max-path-return', self._max_path_return)
        # logger.record_tabular('last-path-return', self._last_path_return)
        logger.record_tabular('avg-path-return', np.mean(self._avg_return))
        # logger.record_tabular('episodes', self._n_episodes)
        logger.record_tabular('total-samples', self._total_samples)


class DummySampler(Sampler):
    def __init__(self, num_envs, **kwargs):
        super(DummySampler, self).__init__(**kwargs)

        """
        DummySampler:
            self._num_envs: number of envs
            
            Three variables below must be initialized by list of size self._num_envs
                self._env
                self._policy
                self._pool
        """

        self._num_envs = num_envs
        self._path_length = np.zeros(self._num_envs)
        self._path_return = np.zeros(self._num_envs)
        self._last_path_return = np.zeros(self._num_envs)
        self._max_path_return = -np.inf * np.ones(self._num_envs)
        self._n_episodes = np.zeros(self._num_envs)
        self._current_observation = None
        self._total_samples = np.zeros(self._num_envs)
        self._max_episodes = 10
        self._arr_return = [deque([], maxlen=self._max_episodes) for _ in range(self._num_envs)]
        self._current_episode_lengths = np.zeros(self._num_envs)

    def sample(self):
        if self._current_observation is None:
            self._current_observation = self.env.reset()

        terminals = [False for _ in range(self._num_envs)]
        path_lengths = np.zeros(self._num_envs)
        for i, (env, policy, pool, current_observation) in enumerate(zip(self.env.envs, self.policy, self.pool, self._current_observation)):
            action, _ = policy.get_action(current_observation)
            next_observation, reward, terminal, info = env.step(action)
            #if i == 0:
            #    print(reward)
            #    print(terminal)
            self._path_length[i] += 1
            self._path_return[i] += reward
            self._total_samples[i] += 1
            path_lengths[i] = self._path_length[i]


            pool.add_sample(
                observation=current_observation,
                action=action,
                reward=reward,
                terminal=terminal,
                next_observation=next_observation)

            if terminal or self._path_length[i] >= self._max_path_length:
                policy.reset()
                self._current_observation[i] = env.reset()
                self._path_length[i] = 0
                self._max_path_return[i] = max(self._max_path_return[i],
                                            self._path_return[i])
                self._last_path_return[i] = self._path_return[i]
                self._arr_return[i].extend([self._path_return[i]])

                self._path_return[i] = 0
                self._n_episodes[i] += 1
                terminals[i] = True
            else:
                self._current_observation[i] = next_observation

        return terminals, path_lengths

    @overrides
    def log_diagnostics(self):
        # super(DummySampler, self).log_diagnostics()
        for i in range(self._num_envs):
            # logger.record_tabular('max-path-return/{i}'.format(i=i), self._max_path_return[i])
            # logger.record_tabular('last-path-return/{i}'.format(i=i), self._last_path_return[i])
            logger.record_tabular('avg-path-return/{i}'.format(i=i), np.mean(self._arr_return[i]))
            # logger.record_tabular('episodes/{i}'.format(i=i), self._n_episodes[i])
            logger.record_tabular('total-samples/{i}'.format(i=i), self._total_samples[i])

    @overrides
    def batch_ready(self):
        enough_samples = [self.pool[i].size >= self._min_pool_size for i in range(self._num_envs)]
        return enough_samples

    def random_batch_with_actor_num(self, actor_num):
        return self.pool[actor_num].random_batch(self._batch_size)

    @overrides
    def terminate(self):
        [self.env.envs[i].terminate() for i in range(self._num_envs)]

