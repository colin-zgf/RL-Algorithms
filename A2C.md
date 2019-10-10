# A2C Algorithm (Proposed in 2016)

## Background

Actor-Critic method is an extension to the Vanilla Policy Gradient (PG) method, which magically improves the stability and convergence speed. It is one of the most powerful methods in Deep Reinforcement Learning. The method's idea of PG is to increase the probability of good actions and decrease the chance of bad ones. In math notation, our PG was written as:

$$\bigtriangledown J \approx \mathbb{E}\begin{bmatrix}Q(s, a)\bigtriangledown log\pi(a|s)\end{bmatrix}\tag{1}$$

The scaling factor $Q(s,a)$ specifies how much we want to increase or decrease the probability of the action taken in the particular state. In the REINFORCE method, we used the discounted total reward as the scaling of the gradient. As an attempt to increase REINFORCE stability, we subtracted the mean reward from the gradient scales. To understand why this helped, let us consider the very simple scenario of an optimization step on which we have three actions with different total discounted rewards: $Q_{1}$, $Q_{2}$ and $Q_{3}$. Now, let us consider the policy gradient with reward to the relative values of thos $Q_{s}$.


## A2C Objective

A2C is a synchronous, deterministic version of A3C; that’s why it is named as “A2C” with the first “A” (“asynchronous”) removed. In A3C each agent talks to the global parameters independently, so it is possible sometimes the thread-specific agents would be playing with policies of different versions and therefore the aggregated update would not be optimal. To resolve the inconsistency, a coordinator in A2C waits for all the parallel actors to finish their work before updating the global parameters and then in the next iteration parallel actors starts from the same policy. The synchronized gradient update keeps the training more cohesive and potentially to make convergence faster.
