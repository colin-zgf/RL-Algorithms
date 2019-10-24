# DQN

## Background

In the Q-learning algorithm, one needs to generate a Q-table with size $S\times A$ where $S$ is the total number of states and $A$ is total number of actions. However, producing and updating a Q-table can become ineffective in big state space environment. Besides, under a lot of situations, some states are rarely used. Thus, there is no need to update Q-table for such kind of states at every iteration.

Instead of using a Qtable, we’ll implement a Neural Network that takes a state and approximates Q-values for each action based on that state.

Threre are some extensions of DQN. The followings show six extensions of DQN

- **N-steps DQN**: How to immprove convergence speed and stability with a simple unrolling of the Bellman equation and why it is not an ultmate solution.

- **Double DQN**: How to deal with DQN overestimation of the value of actions.

- **Noisy Network**: How to make exploration more efficient by adding noise.

- **Prioritized replay buffer**: Why uniform sampling of our experience is not the best way to train.

- **Dueling DQN**: How to improve convergence speed by making our network's architecture closer represent the problem we are solving.

- **Categorical DQN**: How to go beyond the single expected value of action and work with full distributions.

In the rest part, we will focus on those six extenstions of DQN.

## N-Step DQN
