# DDPG (Proposed in 2017)

## DQN Background

DQN has been applied to may situations and achieves success. However, while DQN solves problems with high-dimensional observation spaces, it can only handle
discrete and low-dimensional action spaces. Many tasks of interest, most notably physical control
tasks, have continuous (real valued) and high dimensional action spaces. DQN cannot be straightforwardly applied to continuous domains since it relies on a finding the action that maximizes the
action-value function, which in the continuous valued case requires an iterative optimization process
at every step.
An obvious approach to adapting deep reinforcement learning methods such as DQN to continuous
domains is to to simply discretize the action space. However, this has many limitations, most notably the curse of dimensionality: the number of actions increases exponentially with the number
of degrees of freedom. For example, a 7 degree of freedom system (as in the human arm) with the
coarsest discretization ai 2 f−k; 0; kg for each joint leads to an action space with dimensionality:
37 = 2187. The situation is even worse for tasks that require fine control of actions as they require
a correspondingly finer grained discretization, leading to an explosion of the number of discrete
actions.
