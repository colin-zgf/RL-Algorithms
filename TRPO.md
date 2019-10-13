# TRPO Algorithm (Proposed in 2017)

## Background

Most algorithms for policy optimization can be classified into three broad categories: (1) policy iteration methods, which alternate between estimating the value function under the current policy and improving the policy (2) policy gradient methods, which use an estimator of the gradient of the expected return (total reward) obtained from sample trajectories and have a close connection to policy iteration); and (3) derivative-free optimization methods, such as the cross-entropy method (CEM) and covariance matrix adaptation (CMA), which treat the return as a black box function to be optimized in terms of the policy parameters.

In RL, we optimize a policy $\theta$ for the maximum expected discounted rewards. Nevertheless, there are a few challenges that hurt PG performance. 

$$J_{\theta}= \max \limits_{\tau \sim \pi_{\theta}} \mathbb{E}\begin{bmatrix}\gamma^{t}r_{t}\end{bmatrix}\tag{1}$$

First, PG computes the steepest ascent direction for the rewards **(policy gradient g)** and update the policy towards that direction.

$$\bigtriangledown J_{\theta}=\mathbb{E_{t}}\begin{bmatrix}\gamma^{t}\bigtriangledown_{\theta}log\pi_{\theta}(a_{t}|s_{t})A_{t}\end{bmatrix}\tag{2}$$

However, this method uses the first-order derivative and approximates the surface to be flat. If the surface has high curvature, we can make horrible moves. **Too large of a step leads to a disaster. But, if the step is too small, the model learns too slow.**

Second, it is very hard to have a proper learning rate in RL.

Third, should we constraint the policy changes so we don’t make too aggressive moves? In fact, this is what TRPO does. It limits the parameter changes that are sensitive to the terrain. But providing this solution is not obvious. We adjust the policy by low-level model parameters. To restrict the policy change, what are the corresponding threshold for the model parameters? How can we translate the change in the policy space to the model parameter space?

## TRPO Mechanism

To improve training stability, we should avoid parameter updates that change the policy too much at one step. Trust region policy optimization (TRPO) (Schulman, et al., 2015) carries out this idea by enforcing a KL divergence constraint on the size of policy update at each iteration. This algorithm is similar to natural policy gradient methods and is effective for optimizing large nonlinear policies such as neural networks.

If off policy, the objective function measures the total advantage over the state visitation distribution and actions, while the rollout is following a different behavior policy $\beta(a|s)$:

$$J(θ)=\sum_{s\in S} \rho^{\pi_{\theta_{old}}}  \sum_{a\in A} (\pi_{\theta}(a|s)\hat{A_{\theta_{old}}}(s,a))\tag{3}$$  

$$J(θ)= \sum_{s\in S}\rho^{\pi_{\theta_{old}}}\sum_{a\in A}(\beta (a|s) \frac{\pi_{\theta}(a|s)}{\beta (a|s)}\hat{A_{\theta_{old}}}(s,a))\tag{4}$$ 

$$J(θ)= \mathbb{E_{s \sim \rho^{\pi_{\theta_{old}}}, a \sim \beta}}\begin{bmatrix}\frac{\pi_{\theta}(a|s)}{\beta (a|s)}\hat{A_{\theta_{old}}}(s,a)\end{bmatrix}\tag{5}$$

where $\theta_{old}$ is the policy parameters before the update and thus known to us; $\pi_{\theta_{old}}$ is defined in the same way as above; $\beta(a|s)$ is the behavior policy for collecting trajectories. Noted that we use an estimated advantage $A(.)$ rather than the true advantage function $A(.)$ because the true rewards are usually unknown.

If on policy, the behavior policy is $\pi_{\theta_{old}}(a|s)$:

$$J(θ)= \mathbb{E_{s \sim \rho^{\pi_{\theta_{old}}}, a \sim \pi_{\theta_{old}}}}\begin{bmatrix}\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{old}} (a|s)}\hat{A_{\theta_{old}}}(s,a)\end{bmatrix}\tag{6}$$

TRPO aims to maximize the objective function $J(\theta)$ subject to, trust region constraint which enforces the distance between old and new policies measured by $KL$-divergence to be small enough, within a parameter $\delta$:

$$\mathbb{E_{s \sim \rho^{\pi_{\theta_{old}}}}}\begin{bmatrix}D_{KL}(\pi_{\theta_{old}}(.|s)  || \pi_{\theta} (.|s))\end{bmatrix} \leq \delta \tag{7}$$

Es∼ρπθold[DKL(πθold(.|s)∥πθ(.|s)]≤δ
In this way, the old and new policies would not diverge too much when this hard constraint is met. While still, TRPO can guarantee a monotonic improvement over policy iteration (Neat, right?). Please read the proof in the paper if interested :)
