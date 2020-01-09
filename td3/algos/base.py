import abc
import gtimer as gt
from collections import deque

import numpy as np

from rllab.misc import logger
from rllab.algos.base import Algorithm

from td3.core.serializable import deep_clone
from td3.misc import tf_utils
from td3.misc.sampler import rollouts


class MARLAlgorithm(Algorithm):
    """Abstract Multiple Actor RLAlgorithm (MARLAlgorithm).

    Implements the _train and _evaluate methods to be used
    by classes inheriting from MARLAlgorithm.
    """

    def __init__(
            self,
            sampler,
            n_epochs=1000,
            n_train_repeat=1,
            n_initial_exploration_steps=10000,
            epoch_length=1000,
            eval_n_episodes=10,
            eval_deterministic=True,
            eval_render=False,
            control_interval=1,
            eval_max_episodes=10,
    ):
        """
        Args:
            n_epochs (`int`): Number of epochs to run the training for.
            n_train_repeat (`int`): Number of times to repeat the training
                for single time step.
            n_initial_exploration_steps: Number of steps in the beginning to
                take using actions drawn from a separate exploration policy.
            epoch_length (`int`): Epoch length.
            eval_n_episodes (`int`): Number of rollouts to evaluate.
            eval_deterministic (`int`): Whether or not to run the policy in
                deterministic mode when evaluating policy.
            eval_render (`int`): Whether or not to render the evaluation
                environment.
        """
        self.sampler = sampler

        self._n_epochs = n_epochs
        self._n_train_repeat = n_train_repeat
        self._epoch_length = epoch_length
        self._n_initial_exploration_steps = n_initial_exploration_steps
        self._control_interval = control_interval

        self._eval_n_episodes = eval_n_episodes
        self._eval_deterministic = eval_deterministic
        self._eval_render = eval_render
        self._eval_max_episodes = eval_max_episodes

        self._arr_oldkl = []
        self._arr_bestkl = []

        self._sess = tf_utils.get_default_session()

        self._env = None
        self._arr_actor = None
        self._num_actor = 1
        self._best_actor_num = 0
        self._num_iter_select_best = 1
        self._with_best = False
        self._beta_t = 0
        self._best_flag = np.array([])

        self._save_iter_num = 100000

    def _train(self, env, arr_actor, arr_initial_exploration_policy):
        """Perform RL training.

        Args:
            env (`rllab.Env`): Environment used for training
            policy (`Policy`): Policy used for training
            initial_exploration_policy ('Policy'): Policy used for exploration
                If None, then all exploration is done using policy
            pool (`PoolBase`): Sample pool to add samples to
        """

        self._init_training(env, arr_actor)
        if arr_initial_exploration_policy is None:
            self.sampler.initialize(env, self.policy, self.pool)
            initial_exploration_done = True
        else:
            self.sampler.initialize(env, arr_initial_exploration_policy, self.pool)
            initial_exploration_done = False

        with self._sess.as_default():
            gt.rename_root('RLAlgorithm')
            gt.reset()
            gt.set_def_unique(False)

            for epoch in gt.timed_for(range(self._n_epochs // self._num_actor + 1),
                                      save_itrs=True):
                logger.push_prefix('Epoch #%d | ' % epoch)

                for t in range(self._epoch_length):
                    # TODO.codeconsolidation: Add control interval to sampler
                    if not initial_exploration_done:
                        if self._epoch_length * epoch >= self._n_initial_exploration_steps // self._num_actor:
                            self.sampler.set_policy(self.policy)
                            initial_exploration_done = True

                    self.sampler.sample()
                    if not all(self.sampler.batch_ready()):
                        continue
                    gt.stamp('sample')

                    for i in range(self._n_train_repeat):
                        for j, actor in enumerate(self._arr_actor):
                                self._do_training(
                                    actor=actor,
                                    iteration=(t + epoch * self._epoch_length)*self._n_train_repeat + i,
                                    batch=self.sampler.random_batch_with_actor_num(j))

                    gt.stamp('train')

                logger.record_tabular('beta_t', self._beta_t)
                logger.record_tabular('beta', self._beta_t)
                logger.record_tabular('best_actor_num', self._best_actor_num)
                for i in range(len(self._arr_actor)):
                    logger.record_tabular('mean-okl/{i}'.format(i=i), np.mean(self._arr_oldkl[i]))
                    if self._with_best:
                        logger.record_tabular('mean-bkl/{i}'.format(i=i), np.mean(self._arr_bestkl[i]))


                self._evaluate(epoch)

                # iteration = epoch * self._epoch_length * self._n_train_repeat
                # if iteration % self._save_iter_num == 0 and iteration > 10000:
                #     self.save(iter=iteration)

                params = self.get_snapshot(epoch)
                # logger.save_itr_params(epoch, params)
                times_itrs = gt.get_times().stamps.itrs

                eval_time = times_itrs['eval'][-1] if epoch > 1 else 0
                total_time = gt.get_times().total
                logger.record_tabular('time-eval', eval_time)
                logger.record_tabular('time-total', total_time)
                logger.record_tabular('epoch', epoch * self._num_actor)

                self.sampler.log_diagnostics()

                logger.dump_tabular(with_prefix=False)
                logger.pop_prefix()

                gt.stamp('eval')

            self.sampler.terminate()

    def _evaluate(self, epoch):
        """Perform evaluation for the current policy.

        :param epoch: The epoch number.
        :return: None
        """

        if self._eval_n_episodes < 1:
            return

        for i, (env, actor) in enumerate(zip(self._eval_env, self._arr_actor)):
            policy = actor.policy
            with policy.deterministic(self._eval_deterministic):
                paths = rollouts(env, policy,
                                 self.sampler._max_path_length, self._eval_n_episodes,
                                 )

            total_returns = [path['rewards'].sum() for path in paths]
            episode_lengths = [len(p['rewards']) for p in paths]

            if self._eval_n_episodes > 1:
                logger.record_tabular('return-average/{i}'.format(i=i), np.mean(total_returns))
                logger.record_tabular('episode-length-avg/{i}'.format(i=i), np.mean(episode_lengths))
            else:
                logger.record_tabular('return-average/{i}'.format(i=i), np.mean(total_returns))
                logger.record_tabular('episode-length-avg/{i}'.format(i=i), np.mean(episode_lengths))

            env.log_diagnostics(paths)


        iteration = epoch*self._epoch_length
        # batch = self.sampler.random_batch()
        # self.log_diagnostics(iteration, batch)

    @abc.abstractmethod
    def save(self, iter):
        raise NotImplementedError

    @abc.abstractmethod
    def log_diagnostics(self, actor, iteration, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def get_snapshot(self, epoch):
        raise NotImplementedError

    @abc.abstractmethod
    def _do_training(self, actor, iteration, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def _select_and_copy_best_actor(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _update_old_new(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _update_beta_t(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _init_training(self, env, arr_actor):
        """Method to be called at the start of training.

        :param env: Environment instance.
        :param policy: Policy instance.
        :return: None
        """

        self._env = env
        if self._eval_n_episodes > 0:
            # TODO: This is horrible. Don't do this. Get rid of this.
            import tensorflow as tf
            with tf.variable_scope("low_level_policy", reuse=False):
                self._eval_env = [deep_clone(env_t) for env_t in self._env.envs]
                # self._eval_env = self._env
        self._arr_actor = arr_actor
        self._arr_oldkl = [deque([], maxlen=self._epoch_length) for _ in
                        range(len(self._arr_actor))]
        self._arr_bestkl = [deque([], maxlen=self._epoch_length) for _ in
                        range(len(self._arr_actor))]

    @property
    def policy(self):
        return [actor.policy for actor in self._arr_actor]

    @property
    def env(self):
        return self._env

    @property
    def pool(self):
        return [actor.pool for actor in self._arr_actor]

