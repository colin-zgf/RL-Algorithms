# PPO Algorithm (Proposed in 2017)

## Background

Unlike deep learning, things can become even worse in the RL landscape, as making a bad update of the policy once won't be recovered by subsequent updates. Instead, the bad policy will bring us bad experience samples that we will use on subsequent training steps which could break our policy completely.

It was found that Q-learning (with function approximation) fails on many simple problems and is poorly understood; vanilla policy gradient methods have poor data effiency and robustness; and trust region policy optimization (TRPO) is relatively complicated, and is not compatible with architectures that include noise (such as dropout) or parameter sharing (between the policy and value function, or with auxiliary tasks). Given that TRPO is relatively complicated and we still want to implement a similar constraint, proximal policy optimization (PPO) simplifies it by using a clipped surrogate objective while retaining similar performance. Instead of the gradient of logarithm probability of the action taken, the PPO method uses a different objective: the ratio between the new and the old policy scaled by the advantage.

## PPO Objective

First, let’s denote the probability ratio between old and new policies as:

$$r_{t}(θ)=\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{old}}(a_{t}|s_{t})}\tag{1}$$

In math form, policy gradient methods work by computing an estimator of the policy gradient and plugging it into a stochastic gradient ascent algorithm. The most commonly used gradient estimator has the form below

$$\bigtriangledown J_{\theta}=\mathbb{E_{t}}\begin{bmatrix}\bigtriangledown_{\theta}log\pi_{\theta}(a_{t}|s_{t})A_{t}\end{bmatrix}\tag{2}$$

The new objective function proposed by PPO is 

$$J_{\theta}=\mathbb{E_{t}}\begin{bmatrix}r_{t}(\theta)A_{t}\end{bmatrix}=\mathbb{E_{t}}\begin{bmatrix}\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{old}}(a_{t}|s_{t})}A_{t}\end{bmatrix}\tag{3}$$

Remember that the objective function of TRPO (on policy) is:

$$J^{TRPO}(\theta) = \mathbb{E_{t}}\begin{bmatrix}r_{t}(\theta)A_{\theta_{old}}\end{bmatrix}\tag{4}$$

Without a limitation on the distance between $\theta_{old}$ and $\theta$, to maximize $J^{TRPO}(\theta)$ would lead to instability with extremely large parameter updates and big policy ratios. PPO imposes the constraint by forcing $r(\theta)$ to stay within a small interval around 1, precisely [1-$\varepsilon$, 1+$\varepsilon$], where $\varepsilon$ is a hyperparameter. Then the clipped version of PPO objective is written as: 

$$J^{CLIP}(\theta)=\mathbb{E_{t}}\begin{bmatrix}min(r_{t}(\theta)A_{t}, clip(r_{t}(\theta), 1-\varepsilon, 1+\varepsilon)A_{t})\end{bmatrix}\tag{5}$$

The function $clip(r_{t}(\theta), 1-\varepsilon, 1+\varepsilon)$ clips the ratio within $[1-\varepsilon, 1+\varepsilon]$. The objective function of PPO takes the minimum one between the original value and the clipped version and therefore we lose the motivation for increasing the policy update to extremes for better rewards.

When applying PPO on the network architecture with shared parameters for both policy (actor) and value (critic) functions, in addition to the clipped reward, the objective function is augmented with an error term on the value estimation and an entropy term to encourage sufficient exploration.

$$J^{CLIP'}(\theta)=\mathbb{E}\begin{bmatrix}J^{CLIP}(\theta)−c_{1}(V_{\theta}(s)−V_{target})^2+c_{2}H(s,π_{\theta}(.))\end{bmatrix}\tag{6}$$
where Both $c_{1}$ and $c_{2}$ are two hyperparameter constants.

PPO has been tested on a set of benchmark tasks and proved to produce awesome results with much greater simplicity.

## Generalized Advantage Estimation (GAE)

An n-step look ahead advantage function is defined as:

$$\hat A_{n}^{\pi} = \sum_{t^{'}=t}^{t+n}\gamma^{t^{'}-t}r(s_{t^{'}}, a_{t^{'}})-\hat V_{\phi}^{\pi}(s_{t}) + \gamma^{n}\hat V_{\phi}^{\pi}(s_{t+n})\tag{7}$$

In GAE, we blend the temporal difference results together. Here are different advantage function with 1 to $k$-step lookahead with $\delta_{t}^{v}=r_{t} + \gamma V(s_{t+1}) - V(s_{t})$.

$$\hat A_{t}^{1}:=\delta_{t}^{v}\tag{8}$$

$$\hat A_{t}^{2}:=\delta_{t}^{v} + \gamma \delta_{t+1}^{v}\tag{9}$$

$$\hat A_{t}^{3}:=\delta_{t}^{v} + \gamma \delta_{t+1}^{v} + \gamma^2 \delta_{t+2}^{v}\tag{10}$$

$$\vdot$$

$$\hat A_{t}^{k}:=\sum_{l=0}^{k-1}\delta_{t+1}^{v}\tag{11}$$

## PPO Training

The PPO method uses a slightly different training procedure. When a long sequence of samples is obtained from the environment and then advantage is estimated for the whole sequence, before several epoches of training are performed. **PPO assumes that a large amount of transitions will be obtained from the environment for every subiteration.**
