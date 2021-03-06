# RL-Algorithms

<table>
  <tr>
    <td><img src="images/DDPG_result/ddpg_robots1.gif?raw=true" height="260px" width="400px"></td>
    <td><img src="images/PG_result/pg.gif?raw=true" height="260px" width="400px"></td>
    <td><img src="images/PPO_result/ppo.gif?raw=true" height="260px" width="400px"></td>
  </tr>
</table>  
<table>
  <tr>
    <td><img src="images/DDPG_result/DDPG_arm.gif?raw=true" height="260px" width="400px"></td>
    <td><img src="images/TRPO_result/trpo.gif?raw=true" height="260px" width="400px"></td>
    <td><img src="images/NPG_result/npg.gif?raw=true" height="260px" width="400px"></td>
  </tr>
</table>

## Contact Info
If you have any questions, please leave a message or send the email to 

**guangfeishanghai@sina.com** or **zhuguangfei6699@gmail.com**

## Very Important Note

**If you would like to see all formulas in all .md file, please add the MathJax Plugin. The steps to add the plugin is to search the [Chrome app store](https://chrome.google.com/webstore/category/extensions) and type 'mathjax' in the 'search' section. Then add MathJax Plugin for Github**

## Introduction

In reinforcement learning, a policy can be generated directly from the value function. Besides, a policy can be directly parameterised. This is what the policy gradient comes. If you are not familiar with policy gradient, please read the book ''

## Contents

The following **reinforcement learning** algorithms are included:

- [Q-Learning](https://github.com/colin-zgf/RL-Algorithms/blob/master/Project-Guangfei%20Zhu.pdf)
- [DQN](https://github.com/colin-zgf/RL-Algorithms/blob/master/DQN.md)
- [NPG](https://github.com/colin-zgf/RL-Algorithms/blob/master/NPG.md)
- [A2C](https://github.com/colin-zgf/RL-Algorithms/blob/master/A2C.md)
- [TRPO](https://github.com/colin-zgf/RL-Algorithms/blob/master/TRPO.md)
- [PPO](https://github.com/colin-zgf/RL-Algorithms/blob/master/PPO.md)
- [ACER](https://github.com/colin-zgf/RL-Algorithms/blob/master/ACER.md)
- [DDPG](https://github.com/colin-zgf/RL-Algorithms/blob/master/DDPG.md)
- [D4PG](https://github.com/colin-zgf/RL-Algorithms/blob/master/D4PG.md)
- [SAC](https://github.com/colin-zgf/RL-Algorithms/blob/master/SAC.md)
- [TD3](https://github.com/colin-zgf/RL-Algorithms/blob/master/TD3.md)

## Code

- [Q-Learning](https://github.com/colin-zgf/RL-Algorithms/tree/master/code/Q-learning)
- [Deep-Q network (DQN)](https://github.com/colin-zgf/RL-Algorithms/tree/master/code/DQN)
- [Vanilla Policy Gradient](https://github.com/colin-zgf/RL-Algorithms/tree/master/code/PG_NPG_TRPO_PPO)
- [Natural Policy Gradient (NPG)](https://github.com/colin-zgf/RL-Algorithms/tree/master/code/PG_NPG_TRPO_PPO)
- [Actor-Critic (A2C)](https://github.com/colin-zgf/RL-Algorithms/tree/master/code/A2C)
- [Trust Region Policy Optimization (TRPO)](https://github.com/colin-zgf/RL-Algorithms/tree/master/code/PG_NPG_TRPO_PPO)
- [Proximal Policy Optimization (PPO)](https://github.com/colin-zgf/RL-Algorithms/tree/master/code/PG_NPG_TRPO_PPO)
- [Actor Critic with Experience Replay (ACER)](https://github.com/colin-zgf/RL-Algorithms/tree/master/code/ACER)
- [Deep Deterministic Policy Gradient (DDPG)](https://github.com/colin-zgf/RL-Algorithms/tree/master/code/DDPG)
- [Distributed Distributional Deep Deterministic Policy Gradient (D4PG)](https://github.com/colin-zgf/RL-Algorithms/tree/master/code/D4PG)
- [Soft Actor Critic (SAC)](https://github.com/colin-zgf/RL-Algorithms/tree/master/code/SAC)
- [Twin Delayed Deep Deterministic policy gradient (TD3)](https://github.com/colin-zgf/RL-Algorithms/tree/master/code/TD3)

## Reference

- [Kakade, Sham M. "A natural policy gradient." In Advances in neural information processing systems, pp. 1531-1538. 2002.](https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf)
- [Mnih, Volodymyr, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. "Asynchronous methods for deep reinforcement learning." In International conference on machine learning, pp. 1928-1937. 2016.](https://arxiv.org/pdf/1602.01783.pdf)
- [Schulman, John, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. "Trust region policy optimization." In International conference on machine learning, pp. 1889-1897. 2015.](https://arxiv.org/pdf/1502.05477.pdf)
- [Schulman, John, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).](https://arxiv.org/pdf/1707.06347.pdf)
- [Wang, Ziyu, Victor Bapst, Nicolas Heess, Volodymyr Mnih, Remi Munos, Koray Kavukcuoglu, and Nando de Freitas. "Sample efficient actor-critic with experience replay." arXiv preprint arXiv:1611.01224 (2016).](https://arxiv.org/pdf/1611.01224.pdf)
- [Lillicrap, Timothy Paul, Jonathan James Hunt, Alexander Pritzel, Nicolas Manfred Otto Heess, Tom Erez, Yuval Tassa, David Silver, and Daniel Pieter Wierstra. "Continuous control with deep reinforcement learning." U.S. Patent Application 15/217,758, filed January 26, 2017.](https://arxiv.org/pdf/1509.02971.pdf)
- [Barth-Maron, Gabriel, Matthew W. Hoffman, David Budden, Will Dabney, Dan Horgan, Alistair Muldal, Nicolas Heess, and Timothy Lillicrap. "Distributed distributional deterministic policy gradients." arXiv preprint arXiv:1804.08617 (2018).](https://openreview.net/pdf?id=SyZipzbCb)
- [Haarnoja, Tuomas, Aurick Zhou, Pieter Abbeel, and Sergey Levine. "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor." arXiv preprint arXiv:1801.01290 (2018).](https://arxiv.org/pdf/1801.01290.pdf)
- [Wu, Yuhuai, Elman Mansimov, Roger B. Grosse, Shun Liao, and Jimmy Ba. "Scalable trust-region method for deep reinforcement learning using kronecker-factored approximation." In Advances in neural information processing systems, pp. 5279-5288. 2017.](https://arxiv.org/pdf/1708.05144.pdf)
- [Fujimoto, Scott, Herke van Hoof, and David Meger. "Addressing function approximation error in actor-critic methods." arXiv preprint arXiv:1802.09477 (2018).](https://arxiv.org/pdf/1802.09477.pdf)
- [Lapan, Maxim. Deep Reinforcement Learning Hands-On: Apply modern RL methods, with deep Q-networks, value iteration, policy gradients, TRPO, AlphaGo Zero and more. Packt Publishing Ltd, 2018.](https://books.google.com/books?hl=en&lr=&id=xKdhDwAAQBAJ&oi=fnd&pg=PP1&dq=Deep+Reinforcement+Learning+Hands-On&ots=wTeckp2m8B&sig=cd8CvMkvJMfSb3MMWeHo3VkdYh8#v=onepage&q=Deep%20Reinforcement%20Learning%20Hands-On&f=false)
