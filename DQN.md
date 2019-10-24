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

The basic DQN has a tendency to overestimate values for Q, which may be harmful to training performance and sometimes can lead to suboptimal policies. The root cause of this is the max operation in the Bellman equation, but the strict proof is too complicated to write down here. As a solution to this problem, the authors proposed modifying the Bellman update a bit. And this is where Double DQN comes. The formula of Double DQN is given below:

$$Q(s_{t}, a_{t}) = r_{t} + \gamma \max \limits_{a} Q(s_{t+1}, arg \max \limits_{a} Q(s_{t+1}, a))\tag{4}$$

However, the problem is that we using the same parameters (weights) for estimating the target and the Q value. As a consequence, there is a big correlation between the TD target and the parameters $(w)$ we are changing. Therefore, it means that at every step of training, our Q values shift but also the target value shifts. So, we’re getting closer to our target
but the target is also moving. It’s like chasing a moving target! This lead to a big oscillation in training.

## Noisy Network

Classical DQN achieves exploration by choosing random actions with specially defined hyperparameter epsilon, which is slowly decreased over time from 1.0 (fully random actions) to some small ratio of 0.1 or 0.02. This process works well for simple environments with short episodes, without much non-stationarity during the game, but even in such simple cases, it requires tuning to make training processes efficient.

In noisy network, the author add noise to the weights of fully connected layers of the netwok and adjust the parameters of this noise during training using backpropagation. Two ways of adding the noise

- Independent Gaussian noise.

- Factorized Gaussian noise: To minimize the amount of random values to be sampled, the authors proposed keeping only two random vectors, one with the size of input and another with the size of the output of the layer.

## Prioritized Replay Buffer

This method tries to improve the efficiency of samples in the replay buffer by prioritizing those samples according to the training loss. The SGD method assumes that the data we use for training has a i.i.d. property. To solve this problem, the classic DQN method used a large buffer of transitions, randomly sampled to get the next training batch.

The authors of the paper questioned this uniform random sample policy and proved that by assigning priorities to buffer samples, according to training loss and sampling the buffer proportional to those priorities, we can significantly
improve convergence and the policy quality of the DQN. Thus, the priority is defined as:

$$p(i) = \frac{p_{i}^a}{\sum_{k} p_{k}^a}\tag{4}$$

And during the sampling process, the sample weigts need to be multiplied to the individual sample loss. The value of weight for each sample is defined as:

$$w_{i} = (\frac{1}{N} \cdot \frac{1}{p(i)})^b\tag{5}$$

where $b$ is a hyperparameter between 0 and 1.

## Dueling DQN

The core observation of Dueling DQN lies in the fact that the Q-values Q(s, a) our network is trying to approximate can be divided into quantities: the value of the state V(s) and the advantage of actions in this state A(s, a). 

$$Q(s,a) = V(s) + A(s,a)\tag{5}$$

The
classic DQN network (top) takes features from the convolution layer and, using fully-connected layers, transforms them into a vector of Q-values, one for each action. On the other hand, dueling DQN (bottom) takes convolution features and processes them using two independent paths: one path is responsible for V(s) prediction, which is just a single number, and another path predicts individual advantage values, having the same dimension as Q-values in the classic case. After that, we add V(s) to every value of A(s, a) to obtain the Q(s, a), which is used and trained as normal.

![Dueling DQN](https://github.com/colin-zgf/RL-Algorithms/blob/master/images/DQN_result/dueling/DuelingDQN.png)

## Categorical DQN

The overall idea of categorical DQN is to predict the distribution of value for every action. The authors have shown that the Bellman equation can be generalized for a distribution case and it will have a form.

$$Z(x,a) = R(x,a) + \gamma Z(x^{'}, a^{'})\tag{6}$$

where Z(x,a), R(x,a) are the probability distributions and not numbers.

The resulting distribution can be used to train our network to give better predictions of value distribution for every action of the given state, exactly the same way as with Q-learning. The only difference will be in the loss function, which now has to be replaced to something suitable for distributions' comparison. There are several alternatives available, for example KullbackLeibler (KL)-divergence.

## DQN Experiments

The figure below shows the $\epsilon$ and rewards of N-Step, noisy network, prioritized experience replay and dueling network.

-N-steps
![epsilon](https://github.com/colin-zgf/RL-Algorithms/blob/master/images/DQN_result/n_steps/epsilon.png)
![reward](https://github.com/colin-zgf/RL-Algorithms/blob/master/images/DQN_result/n_steps/reward.png)

-Noisy Network
![](https://github.com/colin-zgf/RL-Algorithms/blob/master/images/DQN_result/noisy/sigma_snr_layer_1.png)
![](https://github.com/colin-zgf/RL-Algorithms/blob/master/images/DQN_result/noisy/sigma_snr_layer_2.png)
![reward](https://github.com/colin-zgf/RL-Algorithms/blob/master/images/DQN_result/noisy/reward.png)

- Prioritized Replay Buffer
![beta](https://github.com/colin-zgf/RL-Algorithms/blob/master/images/DQN_result/prio_replay/beta.png)
![epsilon](https://github.com/colin-zgf/RL-Algorithms/blob/master/images/DQN_result/prio_replay/epsilon.png)
![reward](https://github.com/colin-zgf/RL-Algorithms/blob/master/images/DQN_result/prio_replay/reward.png)

- Dueling Nework
![epsilon](https://github.com/colin-zgf/RL-Algorithms/blob/master/images/DQN_result/dueling/epsilon.png)
![reward](https://github.com/colin-zgf/RL-Algorithms/blob/master/images/DQN_result/dueling/reward.png)
