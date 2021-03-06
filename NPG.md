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

$$D_{KL} (p_{\theta_{old}} \mid \mid p_{\theta}) \approx D_{KL} (p_{\theta_{old}} \mid \mid p_{\theta_{old}}) + d^T \bigtriangledown_{\theta} D_{KL} (p_{\theta_{old}} \mid \mid p_{\theta}) \mid_{\theta=\theta_{old}} + \frac {1}{2} d^T \bigtriangledown_{\theta}^2 D_{KL} (p_{\theta_{old}} \mid \mid p_{\theta})\mid_{\theta=\theta_{old}} d\tag{6}$$

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

Thus,

$$F(\theta_{old}) = \bigtriangledown_{\theta}^2 D_{KL} (p_{\theta_{old}} \mid p_{\theta}) \mid_{\theta=\theta_{old}}\tag{19}$$

Up to now, we will conclude that:

$$D_{KL} (p_{\theta_{old}} \mid p_{\theta}) = \frac{1}{2}d^TF(\theta_{old})d = \frac{1}{2}(\theta-\theta_{old})^TF(\theta_{old})(\theta-\theta_{old})\tag{20}$$

Since KL divergence is roughly analogous to a distance measure between distributions, Fisher information serves as a local distance metric between distributions: how much you change the distribution if you move the parameters a little bit in a given direction.

Substitute for the information matrix in Eqn.(5), one achieves

$$d^*=arg \max \limits_{d} \bigtriangledown_{\theta} J(\theta) \mid_{\theta=\theta_{old}} \cdot d - \frac {1}{2}\lambda (d^T F(\theta_{old}) d)\tag{21}$$

Express the formula in the $min$ format

$$d^*=arg \min \limits_{d} -\bigtriangledown_{\theta} J(\theta) \mid_{\theta=\theta_{old}} \cdot d + \frac {1}{2}\lambda (d^T F(\theta_{old}) d)\tag{22}$$

Setting the gradient to zero:

$$0=\frac {\partial (-\bigtriangledown_{\theta} J(\theta) \mid_{\theta=\theta_{old}} \cdot d + \frac {1}{2}\lambda (d^T F(\theta_{old}) d)}{\partial d} = -\bigtriangledown_{\theta} J(\theta) \mid_{\theta=\theta_{old}} + \frac {1}{2}\lambda ( F(\theta_{old}) d)\tag{23}$$

Thus, 

$$d=\frac{2}{\lambda}F^{-1}(\theta_{old})\bigtriangledown_{\theta} J(\theta) \mid_{\theta=\theta_{old}}\tag{24}$$

Then the natural gradient is:

$$\bigtriangledown_{natural} J(\theta) = F^{-1}(\theta_{old})\bigtriangledown J(\theta)\tag{25}$$

and 

$$\theta_{new}=\theta_{old} + \alpha \cdot F^{-1}(\theta_{old}) \hat g\tag{26}$$

where

$$D_{KL} (\pi_{\theta_{old}} \mid \pi_{\theta}) \approx \frac {1}{2} (\theta-\theta_{old})^T F(\theta_{old})(\theta-\theta_{old})\tag{27}$$

$$\epsilon = \frac {1}{2} (\alpha g_{N})^T F(\alpha g_{N})\tag{28}$$

$$\alpha = \sqrt {\frac{2\epsilon}{g_{N}^T F g_{N}}}\tag{29}$$

## NPG Experiment

The experiment was implemented on the 'HalfCheetah-v2' of Mujuco.

![](https://github.com/colin-zgf/RL-Algorithms/blob/master/images/NPG_result/npg.gif)
