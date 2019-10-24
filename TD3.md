# TD3 (Proposed in 2018)

## Background

In value-based reinforcement learning methods such as deep Q-learning, function approximation errors are known to lead to overestimated value estimates and suboptimal policies. This problem persists in an actor-critic setting and novel mechanisms need to be proposed to minimize its effects on both the actor and the critic. Twin Delayed Deep Deterministic policy gradient (TD3) builds on Double Q-learning, by taking the minimum value between a pair of critics to limit overestimation. The authors draw the connection between target networks and overestimation bias, and suggest delaying policy updates to reduce per-update error and further improve performance.

Overestimation bias is a property of Q-learning in which the maximization of a noisy value estimate induces a consistent
overestimation. In a function approximation setting, this noise is unavoidable given the imprecision of the estimator. This inaccuracy is further exaggerated by the nature of temporal difference learning, in which an estimate of the value function
is updated using the estimate of a subsequent state. This means using an imprecise estimate within each update will lead to an accumulation of error. Due to overestimation bias, this accumulated error can cause arbitrarily bad states to be estimated as high value, resulting in suboptimal policy updates and divergent behavior.

## Mathematics in TD3

In actor-critic methods, the policy or the actor can be updated through the derministic policy gradient algorithm:

$$\bigtriangledown_{phi} J(\phi) = \mathbb{E_{s \sim p_{\pi}}}\begin{bmatrix}\bigtriangledown_{a} Q^{\pi}(s,a)\mid_{a=\pi (s) \bigtriangledown_{\phi} \pi_{\phi} (s)} \end{bmatrix}\tag{1}$$

In Q-learning, the value function can be learned using temporal difference learning. The Bellman equation is a fundamental relationship between the value of a state-action pair $(s, a)$ and the value of the subsequent state-action pair $(s^{'}; a^{'})$:

$$Q^{\pi}(s,a) = r + \gamma \mathbb{E_{s^{'}, a^{'}}} \begin{bmatrix}Q^{\pi}(s^{'}, a^{'}) \end{bmatrix}\tag{2}$$

In deep Q-learning, the network is updated by using temporal difference learning with a secondary frozen target network $Q_{\theta^{'}} (s,a)$ to maintain a fixed objective y over multiple updates:

$$y = r + \gamma Q_{\theta^{'}}(s^{'}, a^{'})\tag{3}$$

where the actions are selected from a target actor network $\pi_{\phi^{'}}$. The weights of a target network are either updated periodically to exactly match the weights of the current network, or by some proportion $\tau$ at each time step:

$$\theta^{'} \leftarrow \tau \theta + (1-\tau)\theta^{'}\tag{4}$$

**Does this theoretical overestimation occur in practice for state-of-the-art methods such as DDPG? The answer is yes!** Clipped Double Q-learning, which greatly reduces overestimation by the critic.

(1) Clipped Double Q-learning: In Double Q-Learning, the action selection and Q-value estimation are made by two networks separately. In the DDPG setting, given two deterministic actors ($\pi_{\phi_{1}}$,$\pi_{\phi_{2}}$) with two corresponding critics ($Q_{\theta_{1}}$,$Q_{\theta_{2}}$), the Double Q-learning Bellman targets look like:

$$y_{1} = r + \gamma Q_{\theta_{2}^{'}} (s^{'}, \pi_{\phi_{1}} (s^{'}))\tag{5}$$

$$y_{2} = r + \gamma Q_{\theta_{1}^{'}} (s^{'}, \pi_{\phi_{2}} (s^{'}))\tag{6}$$
