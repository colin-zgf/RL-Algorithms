# PPO Algorithm (2017)

## Background

Unlike deep learning, things can become even worse in the RL landscape, as making a bad update of the policy once won't be recovered by subsequent updates. Instead, the bad policy will bring us bad experience samples that we will use on subsequent training steps which could break our policy completely.

It was found that Q-learning (with function approximation) fails on many simple problems and is poorly understood; vanilla policy gradient methods have poor data effiency and robustness; and trust region policy optimization (TRPO) is relatively complicated, and is not compatible with architectures that include noise (such as dropout) or parameter sharing (between the policy and value function, or with auxiliary tasks). Given that TRPO is relatively complicated and we still want to implement a similar constraint, proximal policy optimization (PPO) simplifies it by using a clipped surrogate objective while retaining similar performance. Instead of the gradient of logarithm probability of the action taken, the PPO method uses a different objective: the ratio between the new and the old policy scaled by the advantage.

## Algorithm

First, let’s denote the probability ratio between old and new policies as:

$$r_{t}(θ)=\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{old}}(a_{t}|s_{t})}\tag{1}$$

In math form, the classical Asynchronous Advantage Actor-Critic (A3C) objective could be written as

$$\bigtriangledown J_{\theta}=\mathbb{E_{t}}[\bigtriangledown_{\theta}log\pi_{\theta}(a_{t}|s_{t})A_{t}]\tag{2}$$









Then, the objective function of TRPO (on policy) becomes:

JTRPO(θ)=E[r(θ)A^θold(s,a)]
Without a limitation on the distance between θold and θ, to maximize JTRPO(θ) would lead to instability with extremely large parameter updates and big policy ratios. PPO imposes the constraint by forcing r(θ) to stay within a small interval around 1, precisely [1-ε, 1+ε], where ε is a hyperparameter.

JCLIP(θ)=E[min(r(θ)A^θold(s,a),clip(r(θ),1−ϵ,1+ϵ)A^θold(s,a))]
The function clip(r(θ),1−ϵ,1+ϵ) clips the ratio within [1-ε, 1+ε]. The objective function of PPO takes the minimum one between the original value and the clipped version and therefore we lose the motivation for increasing the policy update to extremes for better rewards.

When applying PPO on the network architecture with shared parameters for both policy (actor) and value (critic) functions, in addition to the clipped reward, the objective function is augmented with an error term on the value estimation (formula in red) and an entropy term (formula in blue) to encourage sufficient exploration.

JCLIP'(θ)=E[JCLIP(θ)−c1(Vθ(s)−Vtarget)2+c2H(s,πθ(.))]
where Both c1 and c2 are two hyperparameter constants.

PPO has been tested on a set of benchmark tasks and proved to produce awesome results with much greater simplicity.
