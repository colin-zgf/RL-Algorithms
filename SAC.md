# SAC (Proposed in 2018)

## Background

Model-free deep reinforcement learning (RL) algorithms have been demonstrated on a range of challenging decision making and control tasks. However, these methods typically suffer from two major challenges: very high sample complexity and brittle convergence properties, which necessitate meticulous hyperparameter tuning.

One cause for the poor sample efficiency of deep RL methods is on-policy learning: some of the most commonly used deep RL algorithms, such as TRPO, PPO or A3C, require new samples to be collected for each gradient step. This quickly becomes extravagantly expensive, as the number of gradient steps and samples per step needed to learn an effective policy increases with task complexity.  A commonly used algorithm in such settings, deep deterministic policy gradient (DDPG), provides
for sample-efficient learning but is notoriously challenging to use due to its extreme brittleness and hyperparameter
sensitivity.

Actor-critic algorithms are typically derived starting from policy iteration, which alternates between policy evaluation—computing the value function for a policy—and policy improvement—using the value function to obtain a better policy. In
large-scale reinforcement learning problems, it is typically impractical to run either of these steps to convergence, and
instead the value function and policy are optimized jointly. In this case, the policy is referred to as the actor, and the
value function as the critic. Many actor-critic algorithms build on the standard, on-policy policy gradient formulation
to update the actor, and many of them also consider the entropy of the policy, but instead of
maximizing the entropy, they use it as an regularizer. On-policy training tends to improve stability but results in poor sample complexity.

There have been efforts to increase the sample efficiency while retaining robustness by incorporating off-policy samples and by using higher order variance reduction techniques. However, fully off-policy algorithms still attain better efficiency. A particularly popular off-policy actor-critic method, which is a deep variant of the deterministic policy gradient algorithm,
uses a Q-function estimator to enable off-policy learning, and a deterministic actor that maximizes this Q-function. As such, this method can be viewed both as a deterministic actor-critic algorithm and an approximate Q-learning algorithm. nfortunately, the interplay between the deterministic actor network and the Q-function typically makes DDPG extremely difficult to stabilize and brittle to hyperparameter settings. As a consequence, it is difficult to extend DDPG to complex,
high-dimensional tasks, and on-policy policy gradient methods still tend to produce the best results in such settings. SAC method instead combines off-policy actorcritic training with a stochastic actor, and further aims to maximize the entropy of this actor with an entropy maximization objective. We find that this actually results in a considerably more stable and scalable algorithm that, in practice, exceeds both the efficiency and final performance of DDPG.

SAC is an algorithm which optimizes a stochastic policy in an off-policy way, forming a bridge between stochastic policy optimization and DDPG-style approaches. It isn’t a direct successor to TD3 (having been published roughly concurrently), but it incorporates the clipped double-Q trick, and due to the inherent stochasticity of the policy in SAC, it also winds up benefiting from something like target policy smoothing. 

A central feature of SAC is entropy regularization. The policy is trained to maximize a trade-off between expected return and entropy, a measure of randomness in the policy. This has a close connection to the exploration-exploitation trade-off: increasing entropy results in more exploration, which can accelerate learning later on. It can also prevent the policy from prematurely converging to a bad local optimum. SAC incorporates the entropy measure of the policy into the reward to encourage exploration: we expect to learn a policy that acts as randomly as possible while it is still able to succeed at the task. It is an off-policy actor-critic model following the maximum entropy reinforcement learning framework. A precedent work is Soft Q-learning.

Three key components in SAC:

- An actor-critic architecture with separate policy and value function networks;
- An off-policy formulation that enables reuse of previously collected data for efficiency;
- Entropy maximization to enable stability and exploration.

## Mathematics in SAC

The policy is trained with the objective to maximize the expected return and the entropy at the same time:

$$J(\theta) =  \sum_{t=1}^{T} \mathbb{E_{s_{t}, a_{t} \sim \rho_{\pi_{\theta}}}} \begin{bmatrix}r(s_{t}, a_{t}) + \alpha H(\pi_{\theta}(\cdot \mid s_{t})) \end{bmatrix}\tag{1}$$
 
where H(.) is the entropy measure and α controls how important the entropy term is, known as temperature parameter. The entropy maximization leads to policies that can (1) the policy is incentivized to explore more widely, while giving up on clearly unpromising avenues. (2) the policy can capture multiple modes of nearoptimal behavior. In problem settings where multiple actions seem equally attractive, the policy will commit equal probability mass to those actions.

Precisely, SAC aims to learn three functions:

- The policy with parameter $\theta$, $\pi_{\theta}$.
- Soft Q-value function parameterized by $w$, $Q_{w}$.
- Soft state value function parameterized by $\psi$, $V_{\psi}$; theoretically we can infer $V$ by knowing $Q$ and $\pi$, but in practice, it helps stabilize the training.

Soft Q-value and soft state value according to Bellman equation are defined as:

$$Q(s_{t},a_{t}) = r(s_{t},a_{t}) + \gamma \mathbb{E_{s_{t+1} \sim \rho_{\pi}(s)}} \begin{bmatrix}V(s_{t+1}) \end{bmatrix}\tag{2}$$

$$V(s_{t}) = r(s_{t},a_{t}) + \mathbb{E_{a_{t} \sim \pi}} \begin{bmatrix}Q(s_{t},a_{t}) - \alpha log \pi (a_{t} \mid s_{t}) \end{bmatrix}\tag{3}$$

Thus, 

$$Q(s_{t},a_{t})=r(s_{t},a_{t})+ \mathbb{E_{s_{t+1}, a_{t+1} \sim \rho_{\pi}}} \begin{bmatrix}Q(s_{t+1},a_{t+1}) - \alpha log \pi (a_{t+1} \mid s_{t+1}) \end{bmatrix}\tag{4}$$

where $\rho_{\pi}(s)$ and $\rho_{\pi}(s,a)$ denote the state and the state-action marginals of the state distribution induced by the policy $\pi(a \mid s)$; see the similar definitions in DPG section.

The soft state value function is trained to minimize the mean squared error:

$$J_{V}(\psi) = \mathbb{E_{s_{t} \sim D}}\begin{bmatrix} \frac{1}{2}(V_{\psi}(s_{t})-\mathbb{E} \begin{bmatrix}Q_{w}(s_{t},a_{t}) - \alpha log \pi_{\theta} (a_{t} \mid s_{t}) \end{bmatrix})^2 \end{bmatrix}\tag{5}$$

with gradient:

$$\bigtriangledown_{\psi} J_{V}(\psi)= \bigtriangledown_{\psi} V_{\psi}(s_{t}) (V_{\psi}(s_{t})-Q_{w}(s_{t}, a_{t}) + log \pi_{\theta}(a_{t} \mid s_{t}))\tag{6}$$

where D is the replay buffer.

The soft Q function is trained to minimize the soft Bellman residual:

$$J_{Q}(w) = \mathbb{E_{s_{t}, a_{t} \sim D}}\begin{bmatrix} \frac{1}{2}(Q_{w}(s_{t}, a_{t})-(r(s_{t}, a_{t}) + \gamma \mathbb{E_{s_{t+1} \sim \rho_{\pi}(s)}}\begin{bmatrix}V_{\overline{\psi}}(s_{t+1}) \end{bmatrix})^2\end{bmatrix}\tag{7}$$

with gradient: 

$$\bigtriangledown_{w} J_{Q}(w)= \bigtriangledown_{w} Q_{w}(s_{t}, a_{t}) (Q_{w}(s_{t}, a_{t})-r(s_{t}, a_{t}) - \gamma  V_{\overline{\psi}}(s_{t+1}))\tag{8}$$

where $\overline{\psi}$ is the target value function which is the exponential moving average (or only gets updated periodically in a “hard” way), just like how the parameter of the target Q network is treated in DQN to stabilize the training.

SAC updates the policy to minimize the KL-divergence:

$$\pi_{new} = arg \min \limits_{\pi^{'}\in \Pi} D_{KL}(\pi^{'} (\cdot \mid s_{t}) \mid \mid \frac{exp(Q^{\pi_{old}}(s_{t}, \cdot))}{Z^{\pi_{old}}(s_{t})})\tag{9}$$

$$\pi_{new} = arg \min \limits_{\pi^{'}\in \Pi} D_{KL}(\pi^{'} (\cdot \mid s_{t}) \mid \mid \exp(Q^{\pi_{old}}(s_{t}, \cdot)-log Z^{\pi_{old}}(s_{t})))\tag{10}$$

Objective for update: 

$$J_{\pi}(\theta) = \bigtriangledown_{\theta}D_{KL}(\pi^{'} (\cdot \mid s_{t}) \mid \mid \exp(Q_{w}(s_{t}, \cdot) - log Z_{w}(s_{t}))\tag{11}$$

$$J_{\pi}(\theta) = \mathbb{E_{a_{t} \sim \pi}} \begin{bmatrix}-log \frac{(\exp(Q_{w}(s_{t}, a_{t})-log Z_{w}(s_{t})))}{\pi_{\theta}(a_{t} \mid s_{t})} \end{bmatrix}\tag{12}$$

$$J_{\pi}(\theta) = \mathbb{E_{a_{t} \sim \pi}} \begin{bmatrix}log \pi_{\theta}(a_{t} \mid s_{t}) - Q_{w}(s_{t}, a_{t}) + log Z_{w}(s_{t})  \end{bmatrix}\tag{13}$$

Jπ(θ)=argminπ′∈ΠDKL(π′(.|st)∥exp(Qπold(st,.))Zπold(st))=argminπ′∈ΠDKL(π′(.|st)∥exp(Qπold(st,.)−logZπold(st)))=∇θDKL(πθ(.|st)∥exp(Qw(st,.)−logZw(st)))=Eat∼π[−log(exp(Qw(st,at)−logZw(st))πθ(at|st))]=Eat∼π[logπθ(at|st)−Qw(st,at)+logZw(st)]
where Π is the set of potential policies that we can model our policy as to keep them tractable; for example, Π can be the family of Gaussian mixture distributions, expensive to model but highly expressive and still tractable. Zπold(st) is the partition function to normalize the distribution. It is usually intractable but does not contribute to the gradient. How to minimize Jπ(θ) depends our choice of Π.

This update guarantees that Qπnew(st,at)≥Qπold(st,at), please check the proof on this lemma in the Appendix B.2 in the original paper.

Once we have defined the objective functions and gradients for soft action-state value, soft state value and the policy network, the soft actor-critic algorithm is straightforward:
