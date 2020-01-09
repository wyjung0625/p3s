import numpy as np

from rllab.misc.instrument import VariantGenerator
from td3.misc.utils import flatten, get_git_rev, deep_update

# M = 256
M1 = 400
M2 = 300
REPARAMETERIZE = True


GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'gaussian',
    'reg': 1e-3,
    'action_prior': 'uniform',
    'reparameterize': REPARAMETERIZE
}

GAUSSIAN_POLICY_PARAMS = {
    'hopper': { # 3 DoF
    },
    'half-cheetah': { # 6 DoF
    },
    'walker': { # 6 DoF
    },
    'ant': { # 8 DoF
    },
    'delayed_hopper': { # 3 DoF
    },
    'delayed_half-cheetah': { # 6 DoF
    },
    'delayed_walker': { # 6 DoF
    },
    'delayed_ant': { # 8 DoF
    },

}

DETERMINISTIC_POLICY_PARAMS_BASE = {
    'type': 'deterministic',
    'reg': 1e-3,
    'action_prior': 'uniform',
    'reparameterize': REPARAMETERIZE
}

DETERMINISTIC_POLICY_PARAMS = {
    'hopper': { # 3 DoF
    },
    'half-cheetah': { # 6 DoF
    },
    'walker': { # 6 DoF
    },
    'ant': { # 8 DoF
    },
    'delayed_hopper': {  # 3 DoF
    },
    'delayed_half-cheetah': {  # 6 DoF
    },
    'delayed_walker': {  # 6 DoF
    },
    'delayed_ant': {  # 8 DoF
    },
}

POLICY_PARAMS = {
    'deterministic': {
        k: dict(DETERMINISTIC_POLICY_PARAMS_BASE, **v)
        for k, v in DETERMINISTIC_POLICY_PARAMS.items()
    },
}

VALUE_FUNCTION_PARAMS = {
    'layer_size1': M1,
    'layer_size2': M2,
}

ENV_DOMAIN_PARAMS = {
    'hopper': { # 3 DoF
    },
    'half-cheetah': { # 6 DoF
    },
    'walker': { # 6 DoF
    },
    'ant': { # 8 DoF
    },
    'delayed_hopper': { # 3 DoF
    },
    'delayed_half-cheetah': { # 6 DoF
    },
    'delayed_walker': { # 6 DoF
    },
    'delayed_ant': { # 8 DoF
    },
}

ENV_PARAMS = {
    'hopper': { # 3 DoF
    },
    'half-cheetah': { # 6 DoF
    },
    'walker': { # 6 DoF
    },
    'ant': { # 8 DoF
    },
    'delayed_hopper': { # 3 DoF
    },
    'delayed_half-cheetah': { # 6 DoF
    },
    'delayed_walker': { # 6 DoF
    },
    'delayed_ant': { # 8 DoF
    },
}

ALGORITHM_PARAMS_BASE = {
    'lr': 1e-3,
    'discount': 0.99,
    'policy_update_interval': 2,
    'tau': 0.005,
    'num_actors':4,
    'num_q':2,
    'beta':0.5,
    'with_best':True,
    'reparameterize':True,
    'best_update_interval':1, # 1 means M=250
    'target_ratio':2, # rho

    'base_kwargs': {
        'epoch_length': 1000,
        'n_train_repeat': 1,
        'n_initial_exploration_steps': 1000,
        'eval_render': False,
        'eval_n_episodes': 10,
        'eval_deterministic': True,
    }
}

ALGORITHM_PARAMS = {
    'hopper': { # 3 DoF
        'target_range': 0.05, # d_min
        'base_kwargs': {
            'n_epochs': int(1e3 + 1),
        }
    },
    'half-cheetah': { # 6 DoF
        'target_range': 0.05,  # d_min
        'base_kwargs': {
            'n_epochs': int(1e3 + 1),
            'n_initial_exploration_steps': 10000,
        }
    },
    'walker': { # 6 DoF
        'target_range': 0.02,  # d_min
        'base_kwargs': {
            'n_epochs': int(1e3 + 1),
        }
    },
    'ant': { # 8 DoF
        'target_range': 0.02,  # d_min
        'base_kwargs': {
            'n_epochs': int(1e3 + 1),
            'n_initial_exploration_steps': 10000,
        }
    },
    'delayed_hopper': { # 3 DoF
        'target_range': 0.02,  # d_min
        'base_kwargs': {
            'n_epochs': int(1e3 + 1),
        }
    },
    'delayed_half-cheetah': { # 6 DoF
        'target_range': 0.02,  # d_min
        'base_kwargs': {
            'n_epochs': int(1e3 + 1),
            'n_initial_exploration_steps': 10000,
        }
    },
    'delayed_walker': { # 6 DoF
        'target_range': 0.02,  # d_min
        'base_kwargs': {
            'n_epochs': int(1e3 + 1),
        }
    },
    'delayed_ant': { # 8 DoF
        'target_range': 0.05,  # d_min
        'base_kwargs': {
            'n_epochs': int(1e3 + 1),
            'n_initial_exploration_steps': 10000,
        }
    },
}

NOISE_PARAMS = {
    'exploration_policy_noise_scale':0.1,
    'target_policy_noise_scale':0.2,
    'noise_clip':0.5,
}


REPLAY_BUFFER_PARAMS = {
    'max_replay_buffer_size': 1e6,
}

SAMPLER_PARAMS = {
    'max_path_length': 1000,
    'min_pool_size': 1000,
    'batch_size': 100,
}

RUN_PARAMS_BASE = {
    'seed': [1],
    'snapshot_mode': 'gap',
    'snapshot_gap': 1000,
    'sync_pkl': True,
}

RUN_PARAMS = {
    'hopper': { # 3 DoF
        'snapshot_gap': 600
    },
    'half-cheetah': { # 6 DoF
        'snapshot_gap': 2000
    },
    'walker': { # 6 DoF
        'snapshot_gap': 1000
    },
    'ant': { # 8 DoF
        'snapshot_gap': 2000
    },
    'delayed_hopper': { # 3 DoF
        'snapshot_gap': 600
    },
    'delayed_half-cheetah': { # 6 DoF
        'snapshot_gap': 2000
    },
    'delayed_walker': { # 6 DoF
        'snapshot_gap': 1000
    },
    'delayed_ant': { # 8 DoF
        'snapshot_gap': 2000
    },
}


DOMAINS = [
    'hopper', # 3 DoF
    'half-cheetah', # 6 DoF
    'walker', # 6 DoF
    'ant', # 8 DoF
    'delayed_hopper',  # 3 DoF
    'delayed_half-cheetah',  # 6 DoF
    'delayed_walker',  # 6 DoF
    'delayed_ant',  # 8 DoF
]

TASKS = {
    'hopper': [
        'default',
    ],
    'half-cheetah': [
        'default',
    ],
    'walker': [
        'default',
    ],
    'ant': [
        'default',
    ],
    'delayed_hopper': [
        'default',
    ],
    'delayed_half-cheetah': [
        'default',
    ],
    'delayed_walker': [
        'default',
    ],
    'delayed_ant': [
        'default',
    ],
}

def parse_domain_and_task(env_name):
    domain = env_name # next(domain for domain in DOMAINS if domain in env_name)
    domain_tasks = TASKS[domain]
    task = next((task for task in domain_tasks if task in env_name), 'default')
    return domain, task

def get_variants(domain, task, policy):
    params = {
        'prefix': '{}/{}'.format(domain, task),
        'domain': domain,
        'task': task,
        'git_sha': get_git_rev(),

        'env_params': ENV_PARAMS[domain].get(task, {}),
        'policy_params': POLICY_PARAMS[policy][domain],
        'value_fn_params': VALUE_FUNCTION_PARAMS,
        'algorithm_params': deep_update(
            ALGORITHM_PARAMS_BASE,
            ALGORITHM_PARAMS[domain]
        ),
        'noise_params': NOISE_PARAMS,
        'replay_buffer_params': REPLAY_BUFFER_PARAMS,
        'sampler_params': SAMPLER_PARAMS,
        'run_params': deep_update(RUN_PARAMS_BASE, RUN_PARAMS[domain]),
    }

    # TODO: Remove flatten. Our variant generator should support nested params
    params = flatten(params, separator='.')

    vg = VariantGenerator()
    for key, val in params.items():
        if isinstance(val, list) or callable(val):
            vg.add(key, val)
        else:
            vg.add(key, [val])

    return vg
