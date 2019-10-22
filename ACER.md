# ACER (Proposed in 2017)

## Background

Experience replay has gained popularity in deep Q-learning and is actually a valuable tool for improving sample efficiency. However, we need to do better than deep Q-learning, because it has two important limitations. First, the deterministic nature of the optimal policy limits its use in adversarial domains. Second, finding the greedy action with respect to the Q function is costly for large action spaces.

ACER, short for actor-critic with experience replay, is an off-policy actor-critic model with experience replay, greatly increasing the sample efficiency and decreasing the data correlation. A3C builds up the foundation for ACER, but it is on policy; ACER is A3C’s off-policy counterpart. The major obstacle to making A3C off policy is how to control the stability of the off-policy estimator. ACER proposes three designs to overcome it:

- Use Retrace Q-value estimation;
- Truncate the importance weights with bias correction;
- Apply efficient TRPO.

ACER uses a single deep neural network to estimate the policy $\pi$ and the value function $V$.

## Mathematics in ACER

Retrace is an off-policy return-based $Q$-value estimation algorithm with a nice guarantee for convergence for any target and behavior policy pair ($\pi$, $\beta$), plus good data efficiency.

Given a trajectory generated under the behavior policy $\mu$, the Retrace estimator can be expressed recursively as follows:

$$Q^{ret}(x_{t}, a_{t})=r_{t} + \gamma \overline{\rho_{t+1}}\begin{bmatrix} Q^{ret}(x_{t+1}, a_{t+1}) - Q(x_{t+1}, a_{t+1})\end{bmatrix} + \gamma V(x_{t+1})\tag{1}$$

where $\overline{\rho_{t}}$ is the truncated importance weight, $\overline{\rho_{t}}=min(c, \rho_{t})$ with 

$$\rho_{t} = \frac{\pi (a_{t} \mid x_{t})}{\mu (a_{t} \mid x_{t})}\tag{2}$$

Q is the current value estimate of $Q^{\pi}$, and $V (x) = \mathbb{E_{a \sim \pi}} Q(x, a)$. Retrace is an off-policy, return-based algorithm which has low variance and is proven to converge (in the tabular case) to the value function of the
target policy for any behavior policy. As Retrace uses multistep returns, it can significantly reduce bias in the estimation of the policy gradient.

Recall how TD learning works for prediction:

Compute TD error: 

$$\delta_{t}=R_{t}+\gamma \mathbb{E_{a \sim \pi}}Q(s_{t+1},a_{t+1})−Q(s_{t},a_{t})\tag{3}$$

1. The term $R_{t}+\gamma \mathbb{E_{a \sim \pi}}Q(s_{t+1},a_{t+1})$ is known as “TD target”. The expectation $\mathbb{E_{a \sim \pi}}$ is used because for the future step the best estimation we can make is what the return would be if we follow the current policy $\pi$.

2. Update the value by correcting the error to move toward the goal: $Q(s_{t},a_{t}) \leftarrow Q(s_{t},a_{t})+\alpha \delta_{t}$. In other words, the incremental update on $Q$ is proportional to the TD error: $\Delta Q(s_{t},a_{t})=\alpha \delta_{t}$.

When the rollout is off policy, we need to apply importance sampling on the Q update:

$$\Delta Q^{imp} (s_{t}, a_{t}) = \gamma^{t} \Pi \limits_{1 \le \tau \le t} \frac{\pi(a_{\tau \mid s_{\tau}})}{\beta(a_{\tau \mid s_{\tau}})} \delta_{t}\tag{4}$$


Controlling the variance and stability of off-policy estimators is notoriously hard. Importance sampling is one of the most popular approaches for offpolicy learning
