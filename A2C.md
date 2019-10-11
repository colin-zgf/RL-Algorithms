# A2C Algorithm (Proposed in 2016)

## Background

Actor-Critic method is an extension to the Vanilla Policy Gradient (PG) method, which magically improves the stability and convergence speed. It is one of the most powerful methods in Deep Reinforcement Learning. The method's idea of PG is to increase the probability of good actions and decrease the chance of bad ones. In math notation, our PG was written as:

$$\bigtriangledown J \approx \mathbb{E}\begin{bmatrix}Q(s, a)\bigtriangledown log\pi(a|s)\end{bmatrix}\tag{1}$$

The scaling factor $Q(s,a)$ specifies how much we want to increase or decrease the probability of the action taken in the particular state. In the REINFORCE method, we used the discounted total reward as the scaling of the gradient. As an attempt to increase REINFORCE stability, we subtracted the mean reward from the gradient scales. To understand why this helped, let us consider the very simple scenario of an optimization step on which we have three actions with different total discounted rewards: $Q_{1}$, $Q_{2}$ and $Q_{3}$. Now, let us consider the policy gradient with reward to the relative values of thos $Q_{s}$.

As the first example, let both $Q_{1}$ and $Q_{2}$ be equal to some small positive number and $Q_{3}$ be large negative number. So, actions at the first and second steps led to some small reward, but the third step was not very successful. The resulted combined gradient for all three steps will try to push our policy far from the action at step three and slightly toward the actions taken at step one and two, which is a totally reasonable thing to do.

Now let us imagine that our reward is always positive, only the value is different. This corresponds to adding some constant to all $Q_{1}$, $Q_{2}$ and $Q_{3}$. In this case, $Q_{1}$ and $Q_{2}$ become large positive numbers and $Q_{3}$ will have a small positive value. However, our policy update will become different. Next, we will try hard to push our policy towards actions at the first and second step, and slightly push it towards an action at step three. So, strictly speaking, we are no longer trying to avoid the action taken for step three, despite the fact that the relative rewards are the same.

This dependency of our policy update on the constant added to the reward can slow down our training significantly, as we may require many more samples to average out the effect of such a shift in the PG. Even worse, as our total discounted reward changes over time, with the agent learning how to act better and better, our PG variance could also change. Variance for the version with the baseline is two-to-three orders of magnitude lower than the version without one, which helps the system to converge faster.

## A2C Architecture

A2C is a synchronous, deterministic version of A3C; that’s why it is named as “A2C” with the first “A” (“asynchronous”) removed. In A3C each agent talks to the global parameters independently, so it is possible sometimes the thread-specific agents would be playing with policies of different versions and therefore the aggregated update would not be optimal. To resolve the inconsistency, a coordinator in A2C waits for all the parallel actors to finish their work before updating the global parameters and then in the next iteration parallel actors starts from the same policy. The synchronized gradient update keeps the training more cohesive and potentially to make convergence faster.

Making our baseline state-dependent (which intuitively is a good idea, as different states could have very different baselines) could reduce the variance. Remember in DQN, the total reward itself could be represented as a value of the state plus advantage of the action:

$$Q(s, a) = V(s) + A(s, a)\tag{2}$$

**So why can not we use $V(s)$ as a baseline?** In that case, the scale of our gradient will be just advantage $A(s,a)$, showing how this taken action is better in respect to the average state's value. In fact, we can do this, and it is a very good idea for improving the PG method. The only problem here is: we do not know the value of the $V(s)$ state to subtract it from the discounted total reward $Q(s,a)$. To solve this, let us use another neural network, which will approximate $V(s)$ for every observation. To train it , we can exploit the same training procedure we used in DQN methods: we will carry out the Bellman step and then minimize the mean square error to improve $V(s)$ approximation.

When we know the value for any state (or, at least, have some approximation of it), we can use it to calculate the PG and update our policy network to increase probabilities for actions with good advantage values and decrease the chance of actions with bad advantage. The policy network which returns probability distribution of actions is called the action as it tells us what to do. Another network is called critic, as it allows us to understand how good our actions were.

In practice, policy and value networks partially overlap, mostly due to the efficiency and convergence considerations. In this case, policy and value are implemented as different heads of the network, taking the output from the common body and transforming it into the probability distribution and a single number representing the value of the state. This helps both nework to share low level features such as convolution filters in the Atari agent, but combine them in a different way. The architecture is shown below:


The A2C algorithm is given below

* Initilize network parameters $\theta$ with random values;

* Play N steps in the environment using the current policy $\pi_{\theta}$, saving state $s_{t}$, action $a_{t}$ and reward $r_{t}$;

* $r$=0 if the end of the episode is reached for $V_{\theta}(s_{t})$

* For $i$=$t$-1, $\cdots$, $t_{start}$ (note the steps are processed backwards)

  - Accumulate the PG 
