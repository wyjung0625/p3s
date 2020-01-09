import tensorflow as tf

class Actor(object):
    def __init__(self, actor_num):
        self.actor_num = actor_num
        self.name = "actor{i}".format(i=self.actor_num)
        self._scope_name = (
            tf.get_variable_scope().name + "/" + self.name
        ).lstrip("/")

        self.policy = None
        self.oldpolicy = None
        self.targetpolicy = None
        self.pool = None
        self.eval_pool = None
        self.arr_qf = None
        self.arr_target_qf = None
        self.arr_prior_qf = None
        self.vf = None
        self.target_vf = None
        self.training_ops = list()

    def trainable_params(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self._scope_name)

    def global_params(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self._scope_name)

    def policy_params(self):
        if self.policy is None:
            return []
        with tf.variable_scope(self._scope_name):
            return self.policy.get_params_internal()

    def old_policy_params(self):
        if self.oldpolicy is None:
            return []
        with tf.variable_scope(self._scope_name):
            return self.oldpolicy.get_params_internal()

    def target_policy_params(self):
        if self.targetpolicy is None:
            return []
        with tf.variable_scope(self._scope_name):
            return self.targetpolicy.get_params_internal()

    def qf_params(self):
        if self.arr_qf is None:
            return []
        params = []
        with tf.variable_scope(self._scope_name):
            for qf in self.arr_qf:
                params = params + qf.get_params_internal()
        return params

    def target_qf_params(self):
        if self.arr_target_qf is None:
            return []
        params = []
        with tf.variable_scope(self._scope_name):
            for qf in self.arr_target_qf:
                params = params + qf.get_params_internal()
        return params

    def vf_params(self):
        if self.vf is None:
            return []
        with tf.variable_scope(self._scope_name):
            return self.vf.get_params_internal()

    def target_vf_params(self):
        if self.target_vf is None:
            return []
        with tf.variable_scope(self._scope_name + '/target'):
            return self.target_vf.get_params_internal()

    def current_params(self):
        return self.qf_params() + self.vf_params() + self.policy_params()

    def target_params(self):
        return self.target_qf_params() + self.target_vf_params() + self.target_policy_params()


