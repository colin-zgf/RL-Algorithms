# -*- coding: utf-8 -*-
import time
from datetime import datetime
import gym
import tensorflow as tf


def test(args, counter, test_net, global_net, sess):
    tf.set_random_seed(args.seed)
    env = gym.make(args.env)
    env.seed(args.seed)

    can_test = True  # Test flag
    t_start = 1  # Test step counter to check against global counter
    rewards, steps = [], []  # Rewards and steps for plotting
    l = str(len(str(args.max_training_steps)))  # Max num. of digits for logging steps
    done = True  # Start new episode

    # stores step, reward, avg_steps and time
    results_dict = {'t': [], 'reward': [], 'avg_steps': [], 'time': []}

    while counter.value() <= args.max_training_steps:
        if can_test:
            t_start = counter.value()  # Reset counter

            # Evaluate over several episodes and average results
            avg_rewards, avg_episode_lengths = [], []
            for _ in range(args.evaluation_episodes):
                while True:
                    # Reset or pass on hidden state
                    if done:
                        # Sync with shared model every episode
                        sess.run([tf.assign(t_p, g_p) for t_p, g_p in zip(test_net.a_params, global_net.a_params)])
                        # Reset environment and done flag
                        state = env.reset()
                        done, episode_length = False, 0
                        reward_sum = 0

                    # Optionally render validation states
                    if args.render:
                        env.render()

                    # Choose action greedily
                    action = test_net.choose_action(state)

                    # Step
                    state, reward, done, _ = env.step(action)
                    reward_sum += reward
                    done = done or episode_length >= args.max_episode_length  # Stop episodes at a max length
                    episode_length += 1  # Increase episode counter

                    # Log and reset statistics at the end of every episode
                    if done:
                        avg_rewards.append(reward_sum)
                        avg_episode_lengths.append(episode_length)
                        break

            print(('[{}] Step: {:<' + l + '} Avg. Reward: {:<8} Avg. Episode Length: {:<8}').format(
                datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],
                t_start,
                sum(avg_rewards) / args.evaluation_episodes,
                sum(avg_episode_lengths) / args.evaluation_episodes))

            # storing data in the dictionary.
            results_dict['t'].append(t_start)
            results_dict['reward'].append(sum(avg_rewards) / args.evaluation_episodes)
            results_dict['avg_steps'].append(sum(avg_episode_lengths) / args.evaluation_episodes)
            results_dict['time'].append(str(datetime.now()))

            rewards.append(avg_rewards)  # Keep all evaluations
            steps.append(t_start)
            can_test = False  # Finish testing
        else:
            if counter.value() - t_start >= args.evaluation_interval:
                can_test = True

    time.sleep(0.001)  # Check if available to test every millisecond

    env.close()
