# DDPG (Proposed in 2017)

## DQN Background

Deep Q Network (DQN) has been applied to may situations and achieves success. However, while DQN solves problems with high-dimensional observation spaces, it can only handle discrete and low-dimensional action spaces. Many tasks of interest, most notably physical control tasks, have continuous (real valued) and high dimensional action spaces. DQN cannot be straightforwardly applied to continuous domains since it relies on a finding the action that maximizes the action-value function, which in the continuous valued case requires an iterative optimization process at every step.

An obvious approach to adapting deep reinforcement learning methods such as DQN to continuous domains is to to simply discretize the action space. However, this has many limitations, most notably the curse of dimensionality: the number of actions increases exponentially with the number of degrees of freedom. For example, a 7 degree of freedom system (as in the human arm) with the coarsest discretization $a_{i} \in \begin{bmatrix}−k, 0, k\end{bmatrix}$ for each joint leads to an action space with dimensionality: $3^{7} = 2187$. The situation is even worse for tasks that require fine control of actions as they require a correspondingly finer grained discretization, leading to an explosion of the number of discrete
actions.

Prior to DQN, it was generally believed that learning value functions using large, non-linear function approximators was difficult and unstable. DQN is able to learn value functions using such function approximators in a stable and robust way due to two innovations: 1. the network is trained off-policy with samples from a replay buffer to minimize correlations between samples; 2. the network is trained with a target Q network to give consistent targets during temporal difference backups.

## DPG Background

Deterministic Policy Gradient (DPG) is a variation of the A2C method, but has a very nice property of being off-policy. Deterministic policy gradients also belong to the A2C family, but the policy is deterministic, which means that it directly provides us with the action to take from the state. This makes it possible to apply the chain rule to the $Q$-value, and by maximizing the $Q$, the policy will be improved as well. To understand this, let's look at how the actor and critic are connected in a continuous action domain.

Let's start with an actor, as it is the simpler of the two. What we want from it is the action to take for every given state. In a continuous action domain, every action is a number, so the actor network will take the state as an input and return
$N$ values, one for every action. This mapping will be deterministic, as the same network always returns the same output if the input is the same.

Now let's look at the critic. The role of the critic is to estimate the Q-value, which is a discounted reward of the action taken in some state. However, our action is a vector of numbers, so our critic net now accepts two inputs: the state and the action. The output from the critic will be the single number, which corresponds to the Q-value. This architecture is different from the DQN, when our action space was discrete and, for efficiency, we returned values for all actions in one pass. This mapping is also deterministic. 

So, what do we have? We have two functions, one is the actor, let's call it $\mu (s)$, which converts the state into the action and the other is the critic, by the state and the action giving us the $Q$-value: $Q(s, a)$. We can substitute the actor function into the critic and get the expression with only one input parameter of our state: $Q(s, \mu (s))$. In the end, Neural Networks (NNs) are just functions.

Now the output of the critic gives us the approximation of the entity we're interested in maximizing in the first place: the discounted total reward. This value depends not only on the input state, but also on parameters of the $\theta_{\mu}$ actor and the $\theta_{Q}$ critic networks. At every step of our optimization, we want to change the actor's weights to improve the total reward that we want to get. In mathematical terms, we want the gradient of our policy. In his deterministic policy gradient theorem, David Silver has proved that stochastic policy gradient is equivalent to the deterministic policy gradient. 

## DDPG Architecture

The figure below shows one example of the DDPG actor and critic networks.

![DDPG](https://github.com/colin-zgf/RL-Algorithms/blob/master/images/DDPG_result/DDPG_Architecture.png)

Now, since the policy is deterministic, we have to explore the environment somhow. The simplest method is just to add the random noise to the $\mu (s) + \epsilon N$ actions. A fancier approach to the exploration will be to use the stochastic model. One very popular stochastic process is called OU processes which models the velocity of the massive Brownian particle under the inference of the frction. In a discrete-time case, the OU process could be written as:

$$x_{t+1} = x_{t} + \theta (\mu - x_{t}) + \sigma N\tag{1}$$

where $\theta, \mu, \sigma$ are parameters of the process. This equation expresses the next value generated by the process via the previous value of the noise, adding normal noise $N$. Usually, one could add the value of the OU process to the action return by the actor.

DDPG uses four neural networks:

- Q network: $\theta^Q$
- Deterministic policy network: $\theta^{\mu}$
- Target Q network: $\theta^{Q^{'}}$
- Target policy network: $\theta^{\mu^{'}}$

The Q network and policy network is very much like simple advantage actor-critic, but in DDPG, the actor directly maps states to actions instead of outputting the probability distribution across a discrete action space.

The target networks are time-delayed copies of their original networks that slowly track the learned networks. Using these target value networks greatly improve stability in learning.

**Why do we use experience replay?** In optimization tasks, we want the data to be independently distributed. This fails to be the case when we optimize a sequential decision process in an on-policy way, because the data then would not be independent of each other. When we store them in a replay buffer and take random batches for training, we overcome this issue.

- Actor and Critic Network Updates:

$$y_{i}=r_{i}+\gamma Q^{'}(s_{i+1}, \mu^{'}(s_{i+1} \mid \theta^{\mu^{'}}) \mid \theta^{Q^{'}})\tag{2}$$

In DDPG, the next-state Q values are calculated with the target value network and target policy network. Then we minimize the mean-squared loss between the updated Q value and the orginal Q value:

$$Loss = \frac{1}{N} \sum_{i} (y_{i} - Q(s_{i}, a_{i} \mid \theta^Q))^2\tag{3}$$

For the policy function, the objective is to maximize the expected return:

$$J(\theta) = \mathbb{E} \begin{bmatrix}Q(s,a) \mid_{s=s_{t}, a_{t}=\mu (s)} \end{bmatrix}\tag{4}$$

Then the graident is

$$\bigtriangledown_{\theta^{\mu}} J(\theta) \approx \bigtriangledown_{a} Q(s,a) \bigtriangledown_{\theta^{\mu}} \mu (s \mid \theta^{\mu})\tag{5}$$

DDPG algorithm is shown below:

![DDPG-Algorithm](https://github.com/colin-zgf/RL-Algorithms/blob/master/images/DDPG_result/ddpg_algorithm.png)

## DDPG Experiments

Two experiments were implemented based on DDPG. Both of them are to fectch the object in real-time.

![DDPG1](https://github.com/colin-zgf/RL-Algorithms/blob/master/images/DDPG_result/ddpg_robots1.gif)
![DDPG2](https://github.com/colin-zgf/RL-Algorithms/blob/master/images/DDPG_result/DDPG_arm.gif)
