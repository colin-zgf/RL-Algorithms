import tensorflow as tf
import numpy as np
import gym
from replay_memory import ReplayMemory

ep_num = 0
ep_rewards = []


class ActorCriticNet(object):
    def __init__(self, args, scope, sess=None):
        self.sess = sess
        self.args = args
        self.env = gym.make(args.env).unwrapped
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n
        self.scope = scope
        with tf.variable_scope(self.scope):
            self.s = tf.placeholder(tf.float32, [None, self.input_size], 'states')
            self.a_prob, self.v, self.q, self.a_params, self.c_params = self._build_net(args)
            self.q_ret = tf.placeholder(tf.float32, [None, ], 'target_value')
            self.a_his = tf.placeholder(tf.int32, [None, ], 'history_actions')
            self.rho = tf.placeholder(tf.float32, [None, ], 'importance_sampling')
            self.average_p = tf.placeholder(tf.float32, [None, self.output_size], 'average_policy')
            self.optimizer_a = tf.train.AdamOptimizer(args.learning_rate, name='Optimizer_Actor')
            self.optimizer_c = tf.train.AdamOptimizer(args.learning_rate, name='Optimizer_Critic')

    def _build_net(self, args):
        with tf.variable_scope('actor'):
            l1_actor = tf.layers.dense(self.s, args.num_hidden, tf.nn.tanh,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1), name='l1_actor')
            a_prob = tf.layers.dense(l1_actor, self.output_size, tf.nn.softmax,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1), name='l2_actor')

        with tf.variable_scope('critic'):
            l1_critic = tf.layers.dense(self.s, args.num_hidden, tf.nn.tanh,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1), name='l1_critic')
            q = tf.layers.dense(l1_critic, self.output_size, tf.nn.tanh,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1), name='l2_q')
            v = tf.reduce_sum(q * a_prob)  # V is expectation of Q under π

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/critic')
        return a_prob, v, q, a_params, c_params

    #  choose action greedily
    def choose_action(self, s, greedy=True):
        s = s[np.newaxis, :]
        a_prob = self.sess.run(self.a_prob, {self.s: s})
        if greedy:
            return np.argmax(a_prob.ravel())
        else:
            return np.random.choice(range(a_prob.shape[1]), p=a_prob.ravel())

    def get_policy(self, s):
        s = s[np.newaxis, :]
        a_prob = self.sess.run(self.a_prob, {self.s: s})
        return a_prob.ravel()


class Agent(ActorCriticNet):
    def __init__(self, args, rank, scope, global_net: ActorCriticNet, average_net: ActorCriticNet, sess=None):
        super(Agent, self).__init__(args, scope, sess)
        self.env.seed(args.seed + rank)
        tf.set_random_seed(args.seed + rank)
        self.memory = ReplayMemory(args.memory_capacity, args.max_stored_episode_length)
        self.name = scope
        with tf.variable_scope(self.scope):
            advantage = tf.subtract(self.q_ret, self.v, name='advantage')
            q_a = tf.reduce_sum(self.q * tf.one_hot(self.a_his, self.output_size, dtype=tf.float32), axis=1,
                                keepdims=True)
            # avoid NaN with clipping when value in policy becomes zero, or use
            # neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=l2, labels=self.a_his)
            neg_log_prob = tf.reduce_sum(-tf.log(tf.clip_by_value(self.a_prob, 1e-20, 1.0)) *
                                         tf.one_hot(self.a_his, self.output_size, dtype=tf.float32),
                                         axis=1,
                                         keepdims=True)
            # maximize total reward (log_p * R) is minimize -(log_p * R), tf can only minimize the loss
            self.a_loss = tf.minimum(self.rho, args.c) * tf.reduce_mean(neg_log_prob * advantage)
            bias_weight = tf.maximum((1 - args.c / self.rho), 0) * self.a_prob
            self.a_loss += tf.reduce_sum(bias_weight * -tf.log(tf.clip_by_value(self.a_prob, 1e-20, 1.0)) *
                                         (q_a - self.v))
            # self.a_loss += tf.losses.get_regularization_loss(scope='actor')
            self.kl_loss = -self._kl_div(self.average_p, self.a_prob)
            self.c_loss = tf.reduce_mean(tf.square(self.q_ret - q_a))
            # self.c_loss += tf.losses.get_regularization_loss(scope='critic')

        with tf.name_scope('local_gradients'):
            a_gradients_raw = tf.gradients(self.a_loss, self.a_params)
            kl_gradients = tf.gradients(self.kl_loss, self.a_params)
            # Compute dot products of gradients
            k_dot_g = sum([tf.reduce_sum(k_g * a_g) for k_g, a_g in zip(kl_gradients, a_gradients_raw)])
            k_dot_k = sum([tf.reduce_sum(k_g ** 2) for k_g in kl_gradients])
            # Compute trust region update
            trust_factor = tf.where(tf.equal(k_dot_k, tf.zeros_like(k_dot_k)), 0.0,
                                    tf.maximum(0.0, (k_dot_g - args.delta) / k_dot_k))
            self.a_gradients = [g_p - trust_factor * k_p for g_p, k_p in zip(a_gradients_raw, kl_gradients)]
            self.c_gradients = tf.gradients(self.c_loss, self.c_params)
        with tf.name_scope('pull'):
            self.pull_a_params_op = [tf.assign(l_p, g_p) for l_p, g_p in zip(self.a_params, global_net.a_params)]
            self.pull_c_params_op = [tf.assign(l_p, g_p) for l_p, g_p in zip(self.c_params, global_net.c_params)]
        with tf.name_scope('update'):
            self.accumulate_grads_a = [tf.Variable(tf.zeros_like(para), trainable=False) for para in self.a_params]
            self.accumulate_grads_c = [tf.Variable(tf.zeros_like(para), trainable=False) for para in self.c_params]
            self.zero_ops_actor = [grad.assign(tf.zeros_like(grad)) for grad in self.accumulate_grads_a]
            self.zero_ops_critic = [grad.assign(tf.zeros_like(grad)) for grad in self.accumulate_grads_c]
            self.accumulate_op_a = [tf.assign_add(x, y) for x, y in zip(self.accumulate_grads_a, self.a_gradients)]
            self.accumulate_op_c = [tf.assign_add(x, y) for x, y in zip(self.accumulate_grads_c, self.c_gradients)]
            self.update_a_op = self.optimizer_a.apply_gradients(zip(self.accumulate_grads_a, global_net.a_params))
            self.update_c_op = self.optimizer_c.apply_gradients(zip(self.accumulate_grads_c, global_net.c_params))
            self.update_average_net_op = [tf.assign(a_p, args.alpha * a_p + (1 - args.alpha) * g_p)
                                          for a_p, g_p in zip(average_net.a_params, global_net.a_params)]

    def pull_global(self):
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def update_global(self):
        self.sess.run([self.update_a_op, self.update_c_op])

    def update_average(self):
        self.sess.run(self.update_average_net_op)

    def acer_main(self, counter, coord, average_net: ActorCriticNet):
        while not coord.should_stop() and counter.value() < self.args.max_training_steps:
            self.pull_global()
            self._train(counter, average_net, on_policy=True)
            if len(self.memory) >= self.args.replay_start:
                n = np.random.poisson(self.args.replay_ratio)
                for i in range(n):
                    self._train(counter, average_net, on_policy=False)

    def _train(self, counter, average_net: ActorCriticNet, on_policy: bool):
        global ep_num
        t = 1
        # call on policy part, generate episode and save to replay memory
        if on_policy:
            total_reward = 0
            state = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            while True:
                action = self.choose_action(state, greedy=False)
                policy = self.get_policy(state)
                next_state, reward, is_done, _ = self.env.step(action)
                total_reward += reward
                self.memory.push(state, action, reward, policy, is_done)
                [b.append(item) for b, item in zip((buffer_s, buffer_a, buffer_r),
                                                   (state, action, reward))]
                if t == self.args.t_max or is_done:
                    if is_done:
                        q_ret = 0
                    else:
                        q_ret = self.sess.run(self.v, {self.s: state[np.newaxis, :]})
                    if len(buffer_s) > 2:
                        self.update_net(buffer_s, buffer_a, buffer_r, q_ret, average_net, on_policy)
                    buffer_s, buffer_a, buffer_r = [], [], []

                # Increment counters
                t += 1
                counter.increment()
                state = next_state
                if is_done:
                    ep_num += 1
                    # print('Thread:', self.name, 'Episode:', ep_num, 'Reward:', total_reward)
                    break
        else:
            samples = self.memory.sample(self.args.batch_size)
            buffer_s = [m[0] for m in samples]
            buffer_a = [m[1] for m in samples]
            buffer_r = [m[2] for m in samples]
            buffer_p = [m[3] for m in samples]
            is_done = samples[-1][4]
            if is_done:
                q_ret = 0
            else:
                q_ret = self.sess.run(self.v, {self.s: samples[-1][0][np.newaxis, :]})
            self.update_net(buffer_s, buffer_a, buffer_r, q_ret, average_net, on_policy, buffer_p)

    def update_net(self, buffer_s, buffer_a, buffer_r, q_ret, average_net, on_policy, buffer_p=None):
        self.sess.run([self.zero_ops_actor, self.zero_ops_critic])
        length = len(buffer_r)
        for i in reversed((range(length - 1))):
            if on_policy:
                rho = 1.0
            else:
                pi = self.sess.run(self.a_prob, {self.s: buffer_s[i][np.newaxis, :]})[0][buffer_a[i]]
                mu = buffer_p[i][buffer_a[i]]
                rho = np.nan_to_num(pi / mu)
            average_policy = average_net.get_policy(buffer_s[i])
            q_ret = buffer_r[i] + self.args.gamma * q_ret
            feed_dict = {
                self.s: buffer_s[i][np.newaxis, :],
                self.a_his: np.array([buffer_a[i]]),
                self.q_ret: np.array([q_ret]),
                self.rho: np.array([rho]),
                self.average_p: average_policy[np.newaxis, :]
            }
            self.sess.run([self.accumulate_op_a, self.accumulate_op_c], feed_dict)
            q_i = self.sess.run(self.q, {self.s: buffer_s[i][np.newaxis, :]})[0][buffer_a[i]]
            v_i = self.sess.run(self.v, {self.s: buffer_s[i][np.newaxis, :]})
            q_ret = min(rho, self.args.c) * (q_ret - q_i) + v_i
        self.update_global()
        self.update_average()
        self.pull_global()

    @staticmethod
    def _kl_div(x, y):
        x = tf.distributions.Categorical(probs=x)
        y = tf.distributions.Categorical(probs=y)
        return tf.distributions.kl_divergence(x, y)
