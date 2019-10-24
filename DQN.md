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

To get the idea, let's look at the Bellman update used in Q-learning.

$$Q(s_{t}, a_{t}) = r_{t} + \gamma \max \limits_{a} Q(s_{t+1}, a_{t+1})\tag{1} $$

This equation is recursive, which means that we can express $Q(s_{t+1}, a_{t+1})$ in terms of itself, which gives us this result:

$$Q(s_{t}, a_{t}) = r_{t} + \gamma \max \limits_{a} \begin{bmatrix}r_{a, t+1} +\gamma \max \limits_{a^{'}}Q(s_{t+2}, a^{'}) \end{bmatrix}\tag{2} $$

$$Q(s_{t}, a_{t}) = r_{t} + \gamma r_{t+1} \max \limits_{a^{'}} Q(s_{t+2}, a^{'})\tag{3} $$

This value could be unrolled again and again any number of times. Under certain unrolling steps, it will be benificial for convergence and speed. However, unrolling the Bellman equation many times (e.g. 100 steps ahead) will make DQN fail to converge at all. **Why?** Remember for the $r_{t+1}$ calculation at the beginning of this section, we have omitted the max operation at the intermediate step, assuming that our action selection during experience gathering or our policy was optimal. However, this assumption is mostly not true especially at the beginning of training leading to unsuccessfully training.

## Double DQN

The formula of Double DQN is given below:

$$Q(s_{t}, a_{t}) = r_{t} + \gamma \max \limits_{a} Q(s_{t+1}, arg \max \limits_{a} Q(s_{t+1}, a))\tag{4}$$
