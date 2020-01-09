from numbers import Number

import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc.overrides import overrides
from td3.misc.tf_utils import *


from .base import MARLAlgorithm


class P3S_TD3(MARLAlgorithm, Serializable):
    """Soft Actor-Critic (SAC)

    Example:
    ```python

    env = normalize(SwimmerEnv())

    pool = SimpleReplayPool(env_spec=env.spec, max_pool_size=1E6)

    base_kwargs = dict(
        min_pool_size=1000,
        epoch_length=1000,
        n_epochs=1000,
        batch_size=64,
        scale_reward=1,
        n_train_repeat=1,
        eval_render=False,
        eval_n_episodes=1,
        eval_deterministic=True,
    )

    M = 100
    qf = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M))

    vf = NNVFunction(env_spec=env.spec, hidden_layer_sizes=(M, M))

    policy = GMMPolicy(
        env_spec=env.spec,
        K=2,
        hidden_layers=(M, M),
        qf=qf,
        reg=0.001
    )

    algorithm = SAC(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        pool=pool,
        qf=qf,
        vf=vf,

        lr=3E-4,
        discount=0.99,
        tau=0.01,

        save_full_state=False
    )

    algorithm.train()
    ```

    References
    ----------
    [1] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine, "Soft
        Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
        with a Stochastic Actor," Deep Learning Symposium, NIPS 2017.
    """

    def __init__(
            self,
            base_kwargs,
            env,
            arr_actor,
            best_actor,
            dict_ph,
            arr_initial_exploration_policy,
            with_best = False,
            initial_beta_t = 1,
            plotter=None,
            specific_type=0,

            target_noise_scale=0.2,
            target_noise_clip=0.5,
            target_ratio=2,
            target_range=0.04,
            lr=3e-3,
            discount=0.99,
            tau=0.01,
            policy_update_interval=2,
            best_update_interval=2,
            reparameterize=False,

            save_full_state=False,
    ):
        """
        Args:
            base_kwargs (dict): dictionary of base arguments that are directly
                passed to the base `RLAlgorithm` constructor.

            env (`rllab.Env`): rllab environment object. 
            policy: (`rllab.NNPolicy`): A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.

            qf1 (`valuefunction`): First Q-function approximator.
            qf2 (`valuefunction`): Second Q-function approximator. Usage of two
                Q-functions improves performance by reducing overestimation
                bias.
            vf (`ValueFunction`): Soft value function approximator.

            pool (`PoolBase`): Replay buffer to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.

            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.

            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise. 
            save_full_state (`bool`): If True, save the full class in the
                snapshot. See `self.get_snapshot` for more information.
        """

        Serializable.quick_init(self, locals())
        super(P3S_TD3, self).__init__(**base_kwargs)

        self._env = env
        self._max_actions = int(self._env.action_space.high[0])

        self._arr_actor = arr_actor
        self._best_actor = best_actor
        self._best_actor_num = -1
        self._num_iter_select_best = 1

        assert len(self._env.envs) == len(self._arr_actor)
        self._num_actor = len(self._arr_actor)
        self._n_train_repeat = self._num_actor
        self._dict_ph = dict_ph

        self._arr_initial_exploration_policy = arr_initial_exploration_policy
        self._with_best = with_best
        self._best_flag = np.ones(self._num_actor)
        self._beta_t = initial_beta_t
        self._plotter = plotter

        self._target_noise_scale = target_noise_scale
        self._target_noise_clip = target_noise_clip

        self._target_ratio = target_ratio
        self._target_range = target_range
        self._policy_lr = lr
        self._qf_lr = lr
        self._vf_lr = lr
        self._discount = discount
        self._tau = tau
        self._policy_update_interval = policy_update_interval
        self._best_update_interval = best_update_interval

        # Reparameterize parameter must match between the algorithm and the 
        # policy actions are sampled from.

        self._save_full_state = save_full_state
        self._saver = tf.train.Saver(max_to_keep=1000)
        self._save_dir = '/home/wisrl/wyjung/Result/log/Mujoco/ant_delay20/test_IPE_TD3_NA4_TRatio2_Trange0.03_update1_ver3_new_201906/iter6/'
        # '/test_IPE_TD3_NA' + str(NUM_ACTORS) + '_TRatio' + str(TARGET_RATIO) + '_TRange' + str(
        #     TARGET_RANGE) + '_update' + str(UPDATE_BEST_ITER) + '_ver' + str(VERSION) + '_new_201906'
        self._save_iter_num = 40000

        self._Da = self._env.action_space.flat_dim
        self._Do = self._env.observation_space.flat_dim

        if self._best_actor is not None:
            self._init_critic_update(actor=self._best_actor)
            self._init_actor_update(actor=self._best_actor)
            self._init_target_ops(actor=self._best_actor)

        for actor in self._arr_actor:
            self._init_critic_update(actor=actor)
            self._init_actor_update(actor=actor)
            self._init_target_ops(actor=actor)
            self._init_update_old_new_ops(actor=actor)

        self._sess.run(tf.variables_initializer([
            variable for variable in tf.global_variables()
            if 'low_level_policy' not in variable.name
        ]))

        self._update_old_new()

        for actor in self._arr_actor:
            source_params = actor.current_params()
            target_params = actor.target_params()
            copy_ops = [
                tf.assign(target, source)
                for target, source in zip(target_params, source_params)
            ]

            self._sess.run(copy_ops)

        if self._best_actor is not None:
            source_params = self._best_actor.current_params()
            target_params = self._best_actor.target_params()
            copy_ops = [
                tf.assign(target, source)
                for target, source in zip(target_params, source_params)
            ]

            self._sess.run(copy_ops)

            for actor in self._arr_actor:
                source_params = self._best_actor.trainable_params()
                target_params = actor.trainable_params()

                copy_ops = [
                    tf.assign(target, source)
                    for target, source in zip(target_params, source_params)
                ]

                self._sess.run(copy_ops)

        print("Initialization is finished!")

    @overrides
    def train(self):
        """Initiate training of the SAC instance."""

        self._train(self._env, self._arr_actor, self._arr_initial_exploration_policy)

    def _init_critic_update(self, actor):
        arr_target_qf_t = [target_qf.output_t for target_qf in actor.arr_target_qf]
        min_target_qf_t = tf.minimum(arr_target_qf_t[0], arr_target_qf_t[1])

        ys = tf.stop_gradient(self._dict_ph['rewards_ph'] +
                              (1 - self._dict_ph['terminals_ph']) * self._discount * min_target_qf_t
                              )  # N

        arr_td_loss_t = []
        for qf in actor.arr_qf:
            arr_td_loss_t.append(tf.reduce_mean((ys - qf.output_t)**2))

        td_loss_t = tf.add_n(arr_td_loss_t)
        qf_train_op = tf.train.AdamOptimizer(self._qf_lr).minimize(
            loss=td_loss_t, var_list=actor.qf_params())
        actor.qf_training_ops = qf_train_op
        print('qf params:', actor.qf_params())
        print("target qf param: ", actor.target_qf_params())

    def _init_actor_update(self, actor):
        with tf.variable_scope(actor.name, reuse=tf.AUTO_REUSE):
            qf_t = actor.arr_qf[0].get_output_for(self._dict_ph['observations_ph'], actor.policy.action_t, reuse=tf.AUTO_REUSE)

        actor.oldkl = actor.policy.dist(actor.oldpolicy)

        if self._with_best:
            actor.bestkl = actor.policy.dist(self._best_actor.policy)
            not_best_flag = tf.reduce_sum(self._dict_ph['not_best_ph'] * tf.one_hot(actor.actor_num, self._num_actor))
            policy_kl_loss = tf.reduce_mean(-qf_t) + not_best_flag * self._dict_ph['beta_ph'] * tf.reduce_mean(actor.bestkl)
        else:
            policy_kl_loss = tf.reduce_mean(-qf_t)

        policy_regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES,
            scope=actor.name + '/' + actor.policy.name)

        print("policy regular loss", policy_regularization_losses)

        policy_regularization_loss = tf.reduce_sum(policy_regularization_losses)
        policy_loss = (policy_kl_loss + policy_regularization_loss)

        # We update the vf towards the min of two Q-functions in order to
        # reduce overestimation bias from function approximation error.

        print("policy param: ", actor.policy_params())
        print("old policy param: ", actor.old_policy_params())
        print("target policy param: ", actor.target_policy_params())
        policy_train_op = tf.train.AdamOptimizer(self._policy_lr).minimize(
            loss=policy_loss,
            var_list=actor.policy_params()
        )

        actor.policy_training_ops = policy_train_op

    def _init_target_ops(self, actor):
        source_params = actor.current_params()
        target_params = actor.target_params()

        actor.target_ops = [
            tf.assign(target, (1 - self._tau) * target + self._tau * source)
            for target, source in zip(target_params, source_params)
        ]

    def _init_update_old_new_ops(self, actor):
        source_params = actor.policy_params()
        target_params = actor.old_policy_params()
        actor.copy_old_new_ops = [
            tf.assign(target, source)
            for target, source in zip(target_params, source_params)
        ]

    @overrides
    def _init_training(self, env, arr_actor):
        super(P3S_TD3, self)._init_training(env, arr_actor)
        # self._sess.run([actor.target_ops for actor in self._arr_actor])
        self._best_actor_num = 0
        if self._with_best:
            self._copy_best_actor()


    @overrides
    def _do_training(self, actor, iteration, batch):
        """Runs the operations for updating training and target ops."""

        if iteration > 1 and iteration % self._epoch_length == 0 and actor.actor_num == 0:
            self._update_old_new()

            if self._with_best and iteration % int(self._best_update_interval * self._epoch_length) == 0:
                self._select_best_actor()
                self._best_flag = np.array([int(i == self._best_actor_num) for i in range(len(self._arr_actor))])
                self._copy_best_actor()

        feed_dict = self._get_feed_dict(iteration, batch)
        next_actions = self._get_next_actions(actor, feed_dict)
        feed_dict[self._dict_ph['next_actions_ph']] = next_actions

        self._sess.run(actor.qf_training_ops, feed_dict)

        if iteration % self._policy_update_interval == 0:
            self._sess.run(actor.policy_training_ops, feed_dict)
            self._sess.run(actor.target_ops)

        oldkl_t = self._sess.run(actor.oldkl, feed_dict)
        oldkl_t = np.clip(oldkl_t, 1/10000, 10000)
        self._arr_oldkl[actor.actor_num].extend([np.mean(oldkl_t[np.isfinite(oldkl_t)])])
        if self._with_best:
            bestkl_t = self._sess.run(actor.bestkl, feed_dict)
            bestkl_t = np.clip(bestkl_t, 1/10000, 10000)
            self._arr_bestkl[actor.actor_num].extend([np.mean(bestkl_t[np.isfinite(bestkl_t)])])

        if iteration > 1 and iteration % self._epoch_length == 0 and self._with_best and actor.actor_num == self._num_actor-1:
            self._update_beta_t()

    def _get_next_actions(self, actor, feed_dict):
        actions = np.array(self._sess.run(actor.targetpolicy.action_t, feed_dict))
        noise = np.clip(self._target_noise_scale * np.random.randn(actions.shape[0], actions.shape[1]),
                        -self._target_noise_clip, self._target_noise_clip)
        return np.clip(actions + noise, -self._max_actions, self._max_actions)

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""

        feed_dict = {
            self._dict_ph['observations_ph']: batch['observations'],
            self._dict_ph['actions_ph']: batch['actions'],
            self._dict_ph['next_observations_ph']: batch['next_observations'],
            self._dict_ph['rewards_ph']: batch['rewards'],
            self._dict_ph['terminals_ph']: batch['terminals'],
            self._dict_ph['not_best_ph']: 1-self._best_flag,
            self._dict_ph['beta_ph']: self._beta_t,
        }

        if iteration is not None:
            feed_dict[self._dict_ph['iteration_ph']] = iteration

        return feed_dict

    def _select_best_actor(self):
        mean_returns = [np.mean(self.sampler._arr_return[i]) for i in range(self._num_actor)]
        best_actor_num = np.argmax(mean_returns)
        self._best_actor_num = best_actor_num

    def _copy_best_actor(self):
        source_params = self._arr_actor[self._best_actor_num].policy_params()
        target_params = self._best_actor.policy_params()

        copy_best_ops = [
            tf.assign(target, source)
            for target, source in zip(target_params, source_params)
        ]

        self._sess.run(copy_best_ops)
        print("best actor is copied by the best actor, the actor{i}".format(i=self._best_actor_num))

    def _update_beta_t(self):
        mean_best = []
        mean_old = []
        for i in range(self._num_actor):
            if i == self._best_actor_num:
                continue
            mean_best.append(np.mean(self._arr_bestkl[i]))
            mean_old.append(np.mean(self._arr_oldkl[i]))

        # avg_ratio = np.mean(mean_best) / np.mean(mean_old)

        # D_change = average of mean_old
        # D_best  = average of mean_best
        if np.mean(mean_best) > max(self._target_ratio * np.mean(mean_old), self._target_range) * 1.5:
            if self._beta_t < 1000:
                self._beta_t = self._beta_t * 2
        if np.mean(mean_best) < max(self._target_ratio * np.mean(mean_old), self._target_range) / 1.5:
            if self._beta_t > 1/1000:
                self._beta_t = self._beta_t / 2

        print("next beta_t : ", self._beta_t)

    @overrides
    def _update_old_new(self):
        self._sess.run([actor.copy_old_new_ops for actor in self._arr_actor])

    @overrides
    def save(self, iter):
        save_filename = self._save_dir + 'IPE_TD3_model.ckpt' % int(iter)
        self._saver.save(self._sess, save_filename)
        print("###########" + save_filename + " is saved ##########")

    @overrides
    def log_diagnostics(self, actor, iteration, batch):
        """Record diagnostic information to the logger.

        Records mean and standard deviation of Q-function and state
        value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(iteration, batch)
        min_qf, vf = self._sess.run((actor.min_qf_t, actor.vf_t), feed_dict)

        if self._with_best:
            logger.record_tabular('beta_t', self._beta_t)
            logger.record_tabular('beta', self._beta_t)
            logger.record_tabular('best_actor_num', self._best_actor.actor_num)
            # logger.record_tabular('with_best_flag', self._with_best_flag)
        logger.record_tabular('min-qf-avg/{i}'.format(i=actor.actor_num), np.mean(min_qf))
        logger.record_tabular('min-qf-std/{i}'.format(i=actor.actor_num), np.std(min_qf))
        logger.record_tabular('vf-avg/{i}'.format(i=actor.actor_num), np.mean(vf))
        logger.record_tabular('vf-std/{i}'.format(i=actor.actor_num), np.std(vf))

        actor.policy.log_diagnostics(iteration, batch)
        if self._plotter:
            self._plotter.draw()

    @overrides
    def get_snapshot(self, epoch):
        """Return loggable snapshot of the SAC algorithm.

        If `self._save_full_state == True`, returns snapshot of the complete
        SAC instance. If `self._save_full_state == False`, returns snapshot
        of policy, Q-function, state value function, and environment instances.
        """

        if self._save_full_state:
            snapshot = {
                'epoch': epoch,
                'algo': self
            }
        else:
            snapshot = {
                'epoch': epoch,
            }
            for actor in self._arr_actor:
                snapshot[actor.name + '/policy'] = actor.policy
                snapshot[actor.name + '/target_policy'] = actor.targetpolicy
                for i, qf in enumerate(actor.arr_qf):
                    snapshot[actor.name + '/qf{i}'.format(i=i)] = qf
                for i, target_qf in enumerate(actor.arr_target_qf):
                    snapshot[actor.name + '/target_qf{i}'.format(i=i)] = target_qf

            if self._with_best:
                snapshot['best_actor_num'] = self._best_actor_num
                snapshot['beta_t'] = self._beta_t
                snapshot['beta'] = self._beta_t

        return snapshot

    def __getstate__(self):
        """Get Serializable state of the RLALgorithm instance."""

        d = Serializable.__getstate__(self)
        d.update({
            'actor-qf-params': [[qf.get_param_values() for qf in actor.arr_qf] for actor in self._arr_actor],
            'actor-target-qf-params': [[target_qf.get_param_values() for target_qf in actor.arr_target_qf] for actor in self._arr_actor],
            'actor-policy-params': [actor.policy.get_param_values() for actor in self._arr_actor],
            'actor-target-policy-params': [actor.targetpolicy.get_param_values() for actor in self._arr_actor],
            'actor-pool': [actor.pool.__getstate__() for actor in self._arr_actor],
            'env': [env.__getstate__() for env in self._env.envs],
        })
        return d

    def __setstate__(self, d):
        """Set Serializable state fo the RLAlgorithm instance."""

        Serializable.__setstate__(self, d)
        for i, actor in enumerate(self._arr_actor):
            for j, qf in enumerate(actor.arr_qf):
                qf.set_param_values(d['actor-qf-params'][i][j])
            for j, target_qf in enumerate(actor.arr_target_qf):
                target_qf.set_param_values(d['actor-target-qf-params'][i][j])
            actor.policy.set_param_values(d['actor-policy-params'][i])
            actor.targetpolicy.set_param_values(d['actor-target-policy-params'][i])
            actor.pool.__setstate__(d['actor-pool'][i])
            self._env.envs[i].__setstate__(d['env'][i])
