---
layout: paper
title: FeUdal Networks for Hierarchical Reinforcement Learning
permalink: /papers/feudal-networks-for-hierarchical-learning
paper-author: Vezhnevets et al.
post-author: Federico Taschin
paper-year: 2017
paper-link: https://arxiv.org/abs/1703.01161
---

This paper presents a very interesting approach to Hierarchical Reinforcement Learning. Feudal Networks is a
novel architecture that decouples a Reinforcement Learning agent into a **Manager** and a **Worker**. The
Manager's job is to learn generating useful **goals**, while the Worker learns to achieve them by performing
primitive actions. This allows Manager and Worker to operate at different temporal resolutions, facilitating
the long-term credit assignment problem and inducing the emergence of sub-policies.

## Idea
The agent is split into two modules:
- The **Manager**: Learns an internal representation of the state space and output goals at low temporal
  resolution using a novel architecture, the **Dilated LSTM**.
- The **Worker**: Outputs primitive actions to follow the Manager's goals, and it is trained with an intrinsic
  reward based on the achievement of these goals.

{% include figure.html url="/assets/images/feudal-networks-for-hierarchical-learning/FUN.png"
description="Figure 1. Schematic illustration of FuN. Picture taken from the original paper." %}

### The Manager
The Manager internally computes a latent space representation from which it produces a goal at a lower temporal
resolution. The main components of the Manager are (see Figure 1):
- $s_t \in \mathbb{R}^d$: Latent space representation computed by $\mathcal{f}^{Mspace}$ which, in this paper,
  is a Convolutional  Neural Network.
- $g_t \in \mathbb{R}^d$: The goal vector, produced by a Dilated LSTM $\mathcal{f}^{Mrnn}$, a novel RNN architecture
  which operates at lower temporal resolution than the data stream.

The Manager learns to produce advantageous **directions** (transitions) that the Worker should follow.
The update rule for the Manager is based on a novel concept, the **Transition Policy Gradient**:
\begin{equation}
    \nabla g_t = A_t^M \nabla_{\theta}d_{cos}(s_{t+c} - s_t, g_t(\theta))
\end{equation}
where $A_t^M = R_t - V_t^M(x_t, \theta)$ is the classical Advantage Function computed with an internal critic. More
interesting is the term $\nabla_{\theta} d_{cos}(s_{t+c} - s_t, g_t(\theta))$, that represents the similarity between
the direction $s_{t+c} - s_t$ and the goal direction $g_t(\theta)$. The cosine distance is the cosine between the two
vectors in the $\mathbb{R}^d$ space. Therefore, the more the direction taken by the Worker is aligned with the goal
imposed by the Manager, the more the advantage is considered. The term $c$ acts as the temporal horizon of the Manager.


#### The Worker
The **intrinsic** reward for the Worker is then computed as
\begin{equation} 
    r_t^I = \frac{1}{c \sum_{i=1}^c d_{cos}(s_t - s_{t-i}, g_{t-i})}
\end{equation}
which means that the more the Worker's trajectory is aligned with the goal, the higher the reward will be.
Then, the worker policy can be trained with any Reinforcement Learning algorithm, the paper uses an Advantage Actor
Critic [2]. The Advantage Function used in this case is $A_t^D = (R_t + \alpha R_t^I - V_t^D(x_t, \theta))$ uses
a weighted sum of the intrinsic rewards above and the environment rewards, and it is computed with an internal critic.

## Contribution
- **Long term credit assignment**: By operating at a lower temporal resolution and not being concerned by primitive
  actions, the manager is facilitated in learning long-term consequences of its decisions.
- **Transition Policy Gradient**: The paper introduces this novel concept on which the Manager update step is based
  and provides the relevant theoretical grounds. It shows that this update is way better than simply propagating the
  gradient from the Worker.
- **Dilated LSTM**: The paper introduces a novel RNN architecture and proves its effectiveness in the Fuedal Network
  Architecture.
- **Transfer Learning**: The separation between Manager and Worker allows transfer learning between similar tasks. The
  Transition Policy learned by the Manager is independent from the embodiment of the Worker, and can be transfered 
  between agents with different embodiments.

## Weaknesses
- **Complex implementation**: The presented architecture is composed by many parts, each with its hyperparameters that
  need to be tuned.

## References
[1] Alexander Sasha Vezhnevets, Simon Osindero, Tom Schaul, Nicolas Heess, Max Jaderberg, David Silver,
Koray Kavukcuoglu, DEEP MIND, *FeUdal Networks for Hierarchical Reinforcement Learning*, 2017

[2] Mnih, Volodymyr, Badia, Adria Puigdomenech, Mirza, Mehdi, Graves, Alex, Lillicrap, Timothy P, Harley, Tim,
Silver, David, and Kavukcuoglu, Koray. *Asynchronous methods for deep reinforcement learning* ICML, 2016.
