# A2C Algorithm

Actor-Critic method is an extension to the Vanilla Policy Gradient (PG) method, which magically improves the stability and convergence speed. It is one of the most powerful methods in Deep Reinforcement Learning. The method's idea of PG is to increase the probability of good actions and decrease the chance of bad ones. In math notation, our PG was written as:

$$\bigtriangledown J=\mathbb{E}\begin{bmatrix}Q(s, a)\bigtriangledown log\pi(a|s)\end{bmatrix}\tag{1}$$
