# D4PG (Proposed in 2018)

## Background

For algorithms such as DQN the policy is only implicitly defined in terms of its value function, with actions selected by maximizing this function. In the continuous control domain this would require either a costly optimization step or discretization of the action space. While discretization is perhaps the most straightforward solution, this can prove a particularly poor approximation in highdimensional settings or those that require finer grained control. Instead, a more principled approach is to parameterize the policy explicitly and directly optimize the long term value of following this
policy.

(DDPG) algorithm has several properties that make it ideal for the enhancements, which is at its core an off-policy actor-critic method. In particular, the policy gradient used to update the actor network depends only on a learned critic. This means
that any improvements to the critic learning procedure will directly improve the quality of the actor updates. Due to the fact that DDPG is capable of learning off-policy it is also possible to modify the way in which experience is gathered.

The authors proposed several improvements to the DDPG method we've just seen to improve stability, convergence, and sample efficiency.

- First of all, they adapted the distributional representation of the Q-value. The core idea is to replace a single Q value from the critic with a probability distribution. The Bellman equation is replaced with the Bellman operator, which transforms this distributional representation in a similar way.

- The second improvement was the usage of the n-step Bellman equation, unrolling to speed up the convergence. 

- Another improvement versus the original DDPG method was the usage of the prioritized replay buffer instead of the uniformly sampled buffer. 

The result was impressive: this combination showed the state-of-the art results on the set of continuous control problems.

## D4PG Architecture

1. The most notable change is the critic's output. Instead of returning the single Qvalue for the given state and the action, it now returns $N_ATOMS$ e.g 51 values, corresponding to the probabilities of values from the pre-defined range. 

2. Another difference between D4PG and DDPG is the exploration. DDPG used the OU process for the exploration, but according to D4PG authors, they tried both OU and adding simple random noise to the actions, and the result was the same. So, they used a simpler approach for the exploration in the paper.

3. The last significant difference in the code will be related to the training, as D4PG uses cross-entropy loss to calculate the difference between two probability distributions: returned by the critic and obtained as a result of the Bellman operator. To make both distributions aligned to the same supporting atoms, distribution projection is used.

## Mathematics in D4PG

By making use of the deterministic policy gradient theorem, one can write the gradient of this objective as:

$$(T_{\pi} Q) (x, a) = r(x, a) + \gamma \mathbb{E} \begin{bmatrix}Q(x^{'}, \pi (x^{'})) \mid x, a\end{bmatrix}\tag{1}$$
