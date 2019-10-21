# NPG Algorithm (Proposed in 2002)

## Background

Vanilla Policy Gradient faces challenges including bad sample efficiency and poor convergence. 

- Sample efficiency measures the amount of samples needed to improve a policy. For simulation in the real world, sampling can be expensive.

- Convergence is about whether the policy will converge to an optimal policy.

**Natural Policy Gradient (NPG) converges the model parameters better. It avoids taking bad actions that collapse the training performance** NPG is based on minorize-maximization algorithm which optimizes a policy for the maximum discounted rewards.

The stepwise in gradient descent results from solving the following optimization problem, e.g., using line search:

$$d^*= arg \max \limits_{\begin{Vmatrix} d \end{Vmatrix} \le \epsilon} J (\theta + d)\tag{1}$$

Based on SGD, 

$$\theta_{new} = \theta_{old} + d^*\tag{2}$$

In Natural gradient descent, the stepwise in parameter space is determined by considering the KL divergence in the distributions before and after the update:

$$d^*= arg \max \limits_{d, s.t. KL(\pi_{\theta} \begin{vmatrix} \end{vmatrix} \pi_{\theta+d} ) \le \epsilon} J (\theta + d)\tag{3}$$
