#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/5/23 下午1:19
# @Author  : Yang Yuchi

import argparse
import threading
import multiprocessing as mp
import tensorflow as tf
import gym
from network import ActorCriticNet, Agent
from utils import Counter
from test import test

# Hyper parameters
parser = argparse.ArgumentParser(description='Actor Critic with Experience Replay')
parser.add_argument('--seed', type=int, default=11, help='Random seed')
parser.add_argument('--num-agents', type=int, default=mp.cpu_count(), metavar='N', help='Number of training agents')
parser.add_argument('--max-training-steps', type=int, default=500000, metavar='STEPS', help='Maximum training steps')
parser.add_argument('--t-max', type=int, default=100, metavar='STEPS',
                    help='Max number of forward steps for on-policy learning before update')
parser.add_argument('--max-episode-length', type=int, default=500, metavar='LENGTH', help='Maximum episode length')
parser.add_argument('--replay-start', type=int, default=100, metavar='EPISODES',
                    help='Number of transitions to save before starting off-policy training')
parser.add_argument('--batch-size', type=int, default=100, metavar='SIZE', help='Off-policy batch size')
parser.add_argument('--num-hidden', type=int, default=40, metavar='SIZE', help='Number of hidden neurons')
parser.add_argument('--memory-capacity', type=int, default=20000, metavar='CAPACITY',
                    help='Experience replay memory capacity')
parser.add_argument('--max-stored-episode-length', type=int, default=10000, metavar='LENGTH',
                    help='Maximum length of a stored episode')
parser.add_argument('--replay-ratio', type=int, default=4, metavar='r', help='Ratio of off-policy to on-policy updates')
parser.add_argument('--gamma', type=float, default=0.99, metavar='γ', help='RL discount factor')
parser.add_argument('-c', type=float, default=10, metavar='c', help='Importance weight truncation value')
parser.add_argument('--trust-region', action='store_true', help='Use trust region')
parser.add_argument('--alpha', type=float, default=0.99, metavar='α', help='Average policy decay rate')
parser.add_argument('--delta', type=float, default=1, metavar='δ', help='Trust region threshold value')
parser.add_argument('--learning-rate', type=float, default=0.001, metavar='η', help='Learning rate')
parser.add_argument('--env', type=str, default='CartPole-v1', help='environment name')
parser.add_argument('--evaluation-interval', type=int, default=500, metavar='STEPS',
                    help='Number of training steps between evaluations (roughly)')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N',
                    help='Number of evaluation episodes to average over')
parser.add_argument('--render', action='store_true', help='Render evaluation agent')
parser.add_argument('--output-graph', action='store_true', help='Output Tensor board graph')

args = parser.parse_args()

gym.logger.set_level(gym.logger.ERROR)  # Disable Gym warnings

counter = Counter()

tf.set_random_seed(args.seed)
sess = tf.Session()

with tf.device('/cpu:0'):
    global_net = ActorCriticNet(args, 'global_net', sess)
    average_net = ActorCriticNet(args, 'average_net', sess)
    test_net = ActorCriticNet(args, 'test', sess)
    agents = []
    for i in range(args.num_agents):
        agent_name = 'Agent_%i' % i
        agents.append(Agent(args, i, agent_name, global_net, average_net, sess))

if args.output_graph:
    tf.summary.FileWriter("logs/", sess.graph)

coord = tf.train.Coordinator()
sess.run(tf.global_variables_initializer())
# initialize average network weights to global network weights
sess.run([tf.assign(a_p, g_p) for a_p, g_p in zip(average_net.a_params, global_net.a_params)])


def job():
    agent.acer_main(counter, coord, average_net)


processes = []
p = threading.Thread(target=test, args=(args, counter, test_net, global_net, sess))
p.start()
processes.append(p)

for agent in agents:
    task = threading.Thread(target=job)
    task.start()
    processes.append(task)

coord.join(processes)

