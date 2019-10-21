# NPG Algorithm (Proposed in 2002)

## Background

Vanilla Policy Gradient faces challenges including bad sample efficiency and poor convergence. 

- Sample efficiency measures the amount of samples needed to improve a policy. For simulation in the real world, sampling can be expensive.

- Convergence is about whether the policy will converge to an optimal policy.

**Natural Policy Gradient (NPG) converges the model parameters better. It avoids taking bad actions that collapse the training performance** NPG is based on minorize-maximization algorithm which optimizes a policy for the maximum discounted rewards.

The stepwise in gradient descent results from solving the following optimization problem, e.g., using line search:

$$d*= \max \limits_{\begin{Vmatrix} d \end{Vmatrix} \le \epsilon}} J (\theta + d)\tag{1}$$
