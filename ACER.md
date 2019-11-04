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

Controlling the variance and stability of off-policy estimators is notoriously hard. Importance sampling is one of the most popular approaches for off-policy learning. When the rollout is off policy, we need to apply importance sampling on the Q update:

$$\Delta Q^{imp} (s_{t}, a_{t}) = \gamma^{t}  \frac{\pi(a_{1}\mid s_{1}) \pi(a_{2}\mid s_{2}) \cdots \pi(a_{t}\mid s_{t})}{\beta(a_{1}\mid s_{1}) \beta(a_{2}\mid s_{2}) \cdots \beta(a_{t}\mid s_{t})} \delta_{t}\tag{4}$$

The product of importance weights looks pretty scary when we start imagining how it can cause super high variance and even explode. Retrace $Q$-value estimation method modifies $\Delta Q$ to have importance weights truncated by no more than a constant $c$:

$$\Delta Q^{ret} (s_{t}, a_{t}) = \gamma^{t} min(c, \frac{\pi(a_{1}\mid s_{1}) \pi(a_{2}\mid s_{2}) \cdots \pi(a_{t}\mid s_{t})}{\beta(a_{1}\mid s_{1}) \beta(a_{2}\mid s_{2}) \cdots \beta(a_{t}\mid s_{t})} )\delta_{t}\tag{5}$$

ACER uses $Q^{ret}$ as the target to train the critic by minimizing the $L_{2}$ error term with the gradient: 

$$(Q^{ret}(s,a)−Q(s,a))\bigtriangledown Q(s,a)\tag{6}$$

### Importance Weight Truncation With Bias Correction

To safe-guard against high variance, the author proposes to truncate the importance weights and introduce a correction term. The off-policy ACER gradient:

$$\hat{g_{t}}^{acer} = \overline{\rho_{t}} \bigtriangledown_{\theta} log \pi_{\theta} (a_{t} \mid x_{t}) \begin{bmatrix} Q^{ret}(x_{t}, a_{t}) - V_{\theta_{v}}(x_{t})\end{bmatrix} +$$
$$\mathbb{E_{a \sim \pi}}\begin{bmatrix}\max (0, \frac{\rho_{t}(a) - c}{\rho_{t}(a)}) \bigtriangledown_{\theta} log \pi_{\theta} (a \mid x_{t}) \begin{bmatrix}Q_{\theta_{v}}(x_{t}, a) - V_{\theta_{v}}(x_{t})\end{bmatrix} \end{bmatrix}\tag{7}$$

where $Q_{\theta_{v}}(\cdot)$ and $V_{\theta_{v}}(\cdot)$ are value functions predicted by the critic with parameter $w$. The first term contains the clipped important weight. The clipping helps reduce the variance, in addition to subtracting state value function $V_{\theta_{v}}(\cdot)$ as a baseline. The second term makes a correction to achieve unbiased estimation.

### Efficient Trust Region Optimization

Furthermore, ACER adopts the idea of TRPO but with a small adjustment to make it more computationally efficient: rather than measuring the KL divergence between policies before and after one update, ACER maintains a running average of past policies and forces the updated policy to not deviate far from this average.

ACER is A3C’s off-policy counterpart, so one can set the number of agents in the neural network. The figures below show that 12 agents were created in a certain experiments.

![agent0-3](https://github.com/colin-zgf/RL-Algorithms/blob/master/images/ACER_result/agent0-3.png)
![agent4-7](https://github.com/colin-zgf/RL-Algorithms/blob/master/images/ACER_result/4-7.png)
![agent8-11](https://github.com/colin-zgf/RL-Algorithms/blob/master/images/ACER_result/8-11.png)

