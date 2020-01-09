import argparse
import os
import collections
import sys

import tensorflow as tf
import numpy as np

from rllab.envs.normalized_env import normalize
from td3.envs.vec_env.dummy_vec_env import dummy
from rllab import config

from td3.algos import P3S_TD3
from td3.envs import GymEnv, delay

from td3.misc.instrument import run_td3_experiment
from td3.misc.utils import timestamp, unflatten
from td3.misc.tf_utils import *
from td3.policies import DeterministicPolicy, UniformPolicy
from td3.misc.sampler import DummySampler
from td3.replay_buffers import SimpleReplayBuffer
from td3.value_functions import NNQFunction, NNVFunction
from variants_p3s import parse_domain_and_task, get_variants
from td3.actors.actors import Actor

ENVIRONMENTS = {
    'ant': {
        'default': lambda: GymEnv('Ant-v1'),
    },
    'hopper': {
        'default': lambda: GymEnv('Hopper-v1')
    },
    'half-cheetah': {
        'default': lambda: GymEnv('HalfCheetah-v1')
    },
    'walker': {
        'default': lambda: GymEnv('Walker2d-v1')
    },
    'delayed_ant': {
        'default': lambda: GymEnv('Ant-v1'),
    },
    'delayed_hopper': {
        'default': lambda: GymEnv('Hopper-v1')
    },
    'delayed_half-cheetah': {
        'default': lambda: GymEnv('HalfCheetah-v1')
    },
    'delayed_walker': {
        'default': lambda: GymEnv('Walker2d-v1')
    },
}

DEFAULT_DOMAIN = DEFAULT_ENV = 'ant' #'half-cheetah'
AVAILABLE_DOMAINS = set(ENVIRONMENTS.keys())
AVAILABLE_TASKS = set(y for x in ENVIRONMENTS.values() for y in x.keys())
DELAY_FREQ = 20

# TARGET_RANGE = 0.1      # d_min in the paper
# TARGET_RATIO = 2        # rho in the paper
# UPDATE_BEST_ITER = 1    # UPDATE_BEST_ITER = 1 corresponds to M = 250

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain',
                        type=str,
                        choices=AVAILABLE_DOMAINS,
                        default=None)
    parser.add_argument('--policy',
                        type=str,
                        default='deterministic')
    parser.add_argument('--task',
                        type=str,
                        choices=AVAILABLE_TASKS,
                        default='default')
    parser.add_argument('--env', type=str, default=DEFAULT_ENV)
    parser.add_argument('--exp_name', type=str, default=timestamp())
    parser.add_argument('--mode', type=str, default='local')
    args = parser.parse_args()

    env_name = args.env
    if 'delayed' in args.env:
        env_name = env_name + '_' + str(DELAY_FREQ)
    args.log_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'log', env_name, 'P3S-TD3'))

    return args

def run_experiment(variant):
    env_params = variant['env_params']
    policy_params = variant['policy_params']
    value_fn_params = variant['value_fn_params']
    algorithm_params = variant['algorithm_params']
    replay_buffer_params = variant['replay_buffer_params']
    sampler_params = variant['sampler_params']
    noise_params = variant['noise_params']

    task = variant['task']
    domain = variant['domain']
    print('domain : ', domain)

    num_actors = algorithm_params['num_actors']
    num_q = algorithm_params['num_q']
    with_best = True
    target_range = algorithm_params['target_range']
    target_ratio = algorithm_params['target_ratio']
    best_update_interval = algorithm_params['best_update_interval']

    if 'delayed' in domain:
        print('Delayed!!!!!!!!!!!!!!!')
        env = dummy([delay(normalize(ENVIRONMENTS[domain][task](**env_params)), DELAY_FREQ) for _ in range(num_actors)])
    else:
        env = dummy([normalize(ENVIRONMENTS[domain][task](**env_params)) for _ in range(num_actors)])
    dict_ph = _init_placeholder(env)

    sampler_params['min_pool_size']=algorithm_params['base_kwargs']['n_initial_exploration_steps']

    sampler = DummySampler(num_envs=num_actors, **sampler_params)

    base_kwargs = dict(algorithm_params['base_kwargs'], sampler=sampler)

    pool = SimpleReplayBuffer(env_spec=env.spec, **replay_buffer_params)

    arr_initial_exploration_policy = [UniformPolicy(env_spec=env.spec) for _ in range(num_actors)]

    arr_actor = [Actor(actor_num=i) for i in range(num_actors)]
    for actor in arr_actor:
        init_actor(actor, pool, dict_ph, env, num_q, value_fn_params, noise_params)

    if with_best:
        best_actor = Actor(actor_num=num_actors)
        init_actor(best_actor, pool, dict_ph, env, num_q, value_fn_params, noise_params)
    else:
        best_actor = None

    algorithm = P3S_TD3(
        base_kwargs=base_kwargs,
        env=env,
        arr_actor=arr_actor,
        best_actor=best_actor,
        dict_ph=dict_ph,
        arr_initial_exploration_policy=arr_initial_exploration_policy,
        with_best=with_best,
        target_ratio=target_ratio,
        target_range=target_range,
        lr=algorithm_params['lr'],
        discount=algorithm_params['discount'],
        tau=algorithm_params['tau'],
        reparameterize=algorithm_params['reparameterize'],
        policy_update_interval=algorithm_params['policy_update_interval'],
        best_update_interval=best_update_interval,
        save_full_state=False,
    )

    algorithm._sess.run(tf.global_variables_initializer())

    algorithm.train()


def launch_experiments(variant_generator, args):
    variants = variant_generator.variants()
    # TODO: Remove unflatten. Our variant generator should support nested params
    variants = [unflatten(variant, separator='.') for variant in variants]

    num_experiments = len(variants)
    print('Launching {} experiments.'.format(num_experiments))

    for i, variant in enumerate(variants):
        print("Experiment: {}/{}".format(i, num_experiments))
        run_params = variant['run_params']
        algo_params = variant['algorithm_params']

        experiment_prefix = variant['prefix'] + '/' + args.exp_name
        experiment_name = '{prefix}-{exp_name}-{i:02}'.format(
            prefix=variant['prefix'], exp_name=args.exp_name, i=i)

        run_td3_experiment(
            run_experiment,
            mode=args.mode,
            variant=variant,
            exp_prefix=experiment_prefix,
            exp_name=experiment_name,
            n_parallel=1,
            seed=run_params['seed'],
            terminate_machine=True,
            log_dir=args.log_dir,
            snapshot_mode=run_params['snapshot_mode'],
            snapshot_gap=run_params['snapshot_gap'],
            sync_s3_pkl=run_params['sync_pkl'],
        )


def main():
    args = parse_args()

    domain, task = args.domain, args.task
    if (not domain) or (not task):
        domain, task = parse_domain_and_task(args.env)

    variant_generator = get_variants(domain=domain, task=task, policy=args.policy)
    launch_experiments(variant_generator, args)


def _init_placeholder(env):
    Da = env.action_space.flat_dim
    Do = env.observation_space.flat_dim
    num_actors = env.num_envs

    iteration_ph = get_placeholder(name='iteration', dtype=tf.int64, shape=None)
    observations_ph = get_placeholder(name='observations', dtype=tf.float32, shape=(None, Do))
    next_observations_ph = get_placeholder(name='next_observations', dtype=tf.float32, shape=(None, Do))
    actions_ph = get_placeholder(name='actions', dtype=tf.float32,shape=(None, Da))
    next_actions_ph = get_placeholder(name='next_actions', dtype=tf.float32, shape=(None, Da))
    rewards_ph = get_placeholder(name='rewards', dtype=tf.float32,shape=(None,))
    terminals_ph = get_placeholder(name='terminals', dtype=tf.float32,shape=(None,))
    not_best_ph = get_placeholder(name='not_best', dtype=tf.float32,shape=(num_actors,))
    beta_ph = get_placeholder(name='beta', dtype=tf.float32,shape=None)

    d = {
        'iteration_ph': iteration_ph,
        'observations_ph': observations_ph,
        'next_observations_ph': next_observations_ph,
        'actions_ph': actions_ph,
        'next_actions_ph': next_actions_ph,
        'rewards_ph': rewards_ph,
        'terminals_ph': terminals_ph,
        'not_best_ph': not_best_ph,
        'beta_ph': beta_ph,
    }
    return d


def init_actor(actor, pool, dict_ph, env, num_q, value_fn_params, noise_params):
    M1 = value_fn_params['layer_size1']
    M2 = value_fn_params['layer_size2']
    with tf.variable_scope(actor.name):
        policy = DeterministicPolicy(
            env_spec=env.spec,
            hidden_layer_sizes=(M1, M2),
            reg=1e-3,
            observation_ph=dict_ph['observations_ph'],
            noise_scale=noise_params['exploration_policy_noise_scale'],
        )

        oldpolicy = DeterministicPolicy(
            env_spec=env.spec,
            hidden_layer_sizes=(M1, M2),
            reg=1e-3,
            name='old_deterministic_policy',
            observation_ph=dict_ph['observations_ph'],
            noise_scale=noise_params['exploration_policy_noise_scale'],
        )
        targetpolicy = DeterministicPolicy(
            env_spec=env.spec,
            hidden_layer_sizes=(M1, M2),
            reg=1e-3,
            name='target_deterministic_policy',
            observation_ph=dict_ph['next_observations_ph'],
            noise_scale=noise_params['exploration_policy_noise_scale'],
        )
        actor.policy = policy
        actor.oldpolicy = oldpolicy
        actor.targetpolicy = targetpolicy

        actor.arr_qf = []
        actor.arr_target_qf = []
        for j in range(num_q):
            actor.arr_qf.append(NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M1, M2), name='qf{i}'.format(i=j),
                                            observation_ph=dict_ph['observations_ph'],
                                            action_ph=dict_ph['actions_ph']))
            actor.arr_target_qf.append(NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M1, M2), name='target_qf{i}'.format(i=j),
                                                   observation_ph=dict_ph['next_observations_ph'],
                                                   action_ph=dict_ph['next_actions_ph']))

        actor.pool = pool

if __name__ == '__main__':
    main()
