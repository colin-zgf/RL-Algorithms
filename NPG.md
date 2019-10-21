# NPG Algorithm (Proposed in 2002)

## Background

Vanilla Policy Gradient faces challenges including bad sample efficiency and poor convergence. 

- Sample efficiency measures the amount of samples needed to improve a policy. For simulation in the real world, sampling can be expensive.

- Convergence is about whether the policy will converge to an optimal policy.

**Natural Policy Gradient (NPG) converges the model parameters better. It avoids taking bad actions that collapse the training performance**. NPG is based on minorize-maximization algorithm which optimizes a policy for the maximum discounted rewards.

## Mathematics in NPG

The stepwise in gradient descent results from solving the following optimization problem, e.g., using line search:

$$d^*= arg \max \limits_{\begin{Vmatrix} d \end{Vmatrix} \le \epsilon} J (\theta + d)\tag{1}$$

Based on SGD, 

$$\theta_{new} = \theta_{old} + d^*\tag{2}$$

In Natural gradient descent, the stepwise in parameter space is determined by considering the KL divergence in the distributions before and after the update:

$$d^*= arg \max \limits_{d, s.t. KL(\pi_{\theta} \begin{vmatrix} \end{vmatrix} \pi_{\theta+d} ) \le \epsilon} J (\theta + d)\tag{3}$$

Unconstrained penalized objective:

$$d^*= arg \max \limits_{d} J (\theta + d) - \lambda (D_{KL} (\pi_{\theta} \begin{vmatrix} \end{vmatrix} \pi_{\theta+d} ) - \epsilon)\tag{4}$$

Apply first order Taylor expansion for the loss and second order for the KL, we can get:

$$d^*=arg \max \limits_{d} J (\theta_{old}) + \bigtriangledown_{\theta} J(\theta) \mid_{\theta=\theta_{old}} \cdot d - \frac {1}{2}\lambda (d^T \bigtriangledown_{\theta}^2 (D_{KL} (\pi_{\theta} \begin{vmatrix} \end{vmatrix} \pi_{\theta+d} )\mid_{\theta=\theta_{old}} d) + \lambda \epsilon\tag{5}$$

### Taylor Expansion of KL

$$D_{KL} (p_{\theta_{old}} \mid p_{\theta}) \approx D_{KL} (p_{\theta_{old}} \mid p_{\theta_{old}}) + d^T \bigtriangledown_{\theta} D_{KL} (p_{\theta_{old}} \mid p_{\theta}) \mid_{\theta=\theta_{old}} + \frac {1}{2} d^T \bigtriangledown_{\theta}^2 D_{KL} (p_{\theta_{old}} \mid p_{\theta})\mid_{\theta=\theta_{old}}\tag{6}$$

- Term $\bigtriangledown_{\theta} D_{KL} (p_{\theta_{old}} \mid p_{\theta}) \mid_{\theta=\theta_{old}}$

$$\bigtriangledown_{\theta} D_{KL} (p_{\theta_{old}} \mid p_{\theta}) \mid_{\theta=\theta_{old}} = -\bigtriangledown_{\theta} \mathbb{E_{x \sim p_{\theta_{old}}}} log P_{\theta} (x) \mid_{\theta=\theta_{old}}\tag{7}$$

$$\bigtriangledown_{\theta} D_{KL} (p_{\theta_{old}} \mid p_{\theta}) \mid_{\theta=\theta_{old}} = - \mathbb{E_{x \sim p_{\theta_{old}}}} \bigtriangledown_{\theta}log P_{\theta} (x) \mid_{\theta=\theta_{old}}\tag{8}$$

$$\bigtriangledown_{\theta} D_{KL} (p_{\theta_{old}} \mid p_{\theta}) \mid_{\theta=\theta_{old}} = - \mathbb{E_{x \sim p_{\theta_{old}}}} \frac{1}{P_{\theta_{old}} (x)} \bigtriangledown_{\theta} P_{\theta} (x) \mid_{\theta=\theta_{old}}\tag{9}$$

$$\bigtriangledown_{\theta} D_{KL} (p_{\theta_{old}} \mid p_{\theta}) \mid_{\theta=\theta_{old}} = - \int_{x} P_{\theta_{old}} (x) \frac{1}{P_{\theta_{old}} (x)} \bigtriangledown_{\theta} P_{\theta} (x) \mid_{\theta=\theta_{old}}\tag{10}$$

$$\bigtriangledown_{\theta} D_{KL} (p_{\theta_{old}} \mid p_{\theta}) \mid_{\theta=\theta_{old}} = - \int_{x} \bigtriangledown_{\theta} P_{\theta} (x) \mid_{\theta=\theta_{old}}\tag{11}$$

$$\bigtriangledown_{\theta} D_{KL} (p_{\theta_{old}} \mid p_{\theta}) \mid_{\theta=\theta_{old}} = - \bigtriangledown_{\theta} \int_{x}  P_{\theta} (x) \mid_{\theta=\theta_{old}} = 0\tag{12}$$

- Term $\bigtriangledown_{\theta}^2 D_{KL} (p_{\theta_{old}} \mid p_{\theta})\mid_{\theta=\theta_{old}}$

$$\bigtriangledown_{\theta}^2 D_{KL} (p_{\theta_{old}} \mid p_{\theta})\mid_{\theta=\theta_{old}} = - \mathbb{E_{x \sim p_{\theta_{old}}}} \bigtriangledown_{\theta}^2 log P_{\theta} (x) \mid_{\theta=\theta_{old}}\tag{13}$$

$$\bigtriangledown_{\theta}^2 D_{KL} (p_{\theta_{old}} \mid p_{\theta})\mid_{\theta=\theta_{old}} = - \mathbb{E_{x \sim p_{\theta_{old}}}} \bigtriangledown_{\theta} (\frac{\bigtriangledown_{\theta} P_{\theta} (x)}{P_{\theta} (x)})  \mid_{\theta=\theta_{old}}\tag{14}$$

$$\bigtriangledown_{\theta}^2 D_{KL} (p_{\theta_{old}} \mid p_{\theta})\mid_{\theta=\theta_{old}} = - \mathbb{E_{x \sim p_{\theta_{old}}}} (\frac{\bigtriangledown_{\theta}^2 P_{\theta} (x) P_{\theta} (x) - \bigtriangledown_{\theta} P_{\theta} (x)\bigtriangledown_{\theta} P_{\theta} (x)^T}{P_{\theta} (x)^2})  \mid_{\theta=\theta_{old}}\tag{15}$$

$$\bigtriangledown_{\theta}^2 D_{KL} (p_{\theta_{old}} \mid p_{\theta})\mid_{\theta=\theta_{old}} = - \mathbb{E_{x \sim p_{\theta_{old}}}} \frac{\bigtriangledown_{\theta}^2 P_{\theta} (x) \mid_{\theta=\theta_{old}}}{P_{\theta_{old}}(x)} +  \mathbb{E_{x \sim p_{\theta_{old}}}} \bigtriangledown_{\theta} log P_{\theta} (x) \bigtriangledown_{\theta} log P_{\theta} (x)^T\mid_{\theta=\theta_{old}}\tag{16}$$


$$\bigtriangledown_{\theta}^2 D_{KL} (p_{\theta_{old}} \mid p_{\theta})\mid_{\theta=\theta_{old}} = \mathbb{E_{x \sim p_{\theta_{old}}}} \bigtriangledown_{\theta} log P_{\theta} (x) \bigtriangledown_{\theta} log P_{\theta} (x)^T\mid_{\theta=\theta_{old}}\tag{17}$$

The Fisher Information Matrix (FIM) is defined as:

$$F(\theta) = \mathbb{E_{\theta}} \begin{bmatrix}\bigtriangledown_{\theta} log P_{\theta} (x) \bigtriangledown_{\theta} log P_{\theta} (x)^T\end{bmatrix}\tag{18}$$
