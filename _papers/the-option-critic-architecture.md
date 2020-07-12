---
layout: paper
title: The Option Critic Architecture
category: hierarchical
permalink: /papers/the-option-critic-architecture
paper-author: Pierre-Luc Bacon, Jean Harb, Doina Precup
post-author: Federico Taschin
paper-year: 2016
paper-link: https://dl.acm.org/doi/10.5555/3298483.3298491
---
<!--
Disclaimer and authorship:
This article is provided for free only for your personal informational and entertainment purposes. No commercial use of it is allowed.

Please note there might be mistakes. We would be grateful to receive (constructive) criticism if you spot any. You can reach us at: ai.campus.ai@gmail.com or directly open an issue on our github repo: https://github.com/CampusAI/CampusAI.github.io

If considering to use the text please cite the original author/s of the lecture/paper.
Furthermore, please acknowledge our work by adding a link to our website: https://campusai.github.io/ and citing our names: Oleguer Canal and Federico Taschin.
-->

The Options framework provides theoretical grounds for temporal abstraction in Reinforcement Learning.
Each Option can be considered as a macro-action with its policy and termination condition, leading
to two levels of policies: one *policy over options* and several *intra-option policies*. This paper
presents the Option-Critic Architecture, capable of learning both intra-option policies and termination
conditions while executing the policy over options, without the need of subgoals or additional rewards.

## Idea

#### The Option Framework
In the Options framework, an option $w \in \Omega$ is a macro-action that, once selected, performs actions
according to its internal policy. It is composed by:
- A policy $\pi_w(s, a)$ that represents the probability of taking action $a$ in state $s$ while
  executing option $w$.
- An initiation set $\mathcal{I}_w \in S$ that defines the states in which an option can be initiated.
- A termination function $\beta_w(s)$ that represents the probability of option $w$ terminating in
  state $s$.

It is, therefore, possible to define, similarly to the standard Reinforcement Learning framework, an
option-value function $Q_{\Omega}(s, w)$, representing the value of being in state $s$ performing
option $w$.

For primitive actions, it is defined $Q_U(s, w, a)$, that represents the value of performing action $a$
in the state-option pair $s, w$. 

#### Optimizing $\pi_w$ and $\beta_w$
By employing differentiable intra-option policies $\pi_{w, \theta}(s, a)$ and termination functions
$\beta_{w, \vartheta}(s)$, parametrized respectively by $\theta$ and $\vartheta$, the authors derive the
gradient of $Q_{\Omega}(s, w)$ with respect to $\theta$ and $\vartheta$. Having the gradients allows performing
gradient updates on both the intra-option policy parameters $\theta$ and the termination parameters $\vartheta$.

The Option-Critic Architecture approximates the value of $Q_U$ and updates the intra-option policies and
termination functions with the gradient step on $\theta$ and $\vartheta$. Algorithm 1 shows an example of
this architecture. Note that values should be updated at a higher rate, while policies and termination
functions should be updated less frequently. In this example, the $Q_U$ function is updated in a tabular
way with Q-Learning.

{% include figure.html url="/assets/images/the-option-critic-architecture/algo1optioncritic.png"
description="Learning intra-option policies and termination functions with the Option Critic Architecture" %}

It is important to note that Algorithm 1 provides only an example implementation of the Option-Critic
Architecture, but many techniques can be employed to estimate $Q_U$ and $Q_{\Omega}$.

## Contribution
 - **Learn without additional help**: The Option-Critic architecture allows us to learn a policy over options,
   the intra-option policies, and the termination functions without specifying option sub-goals or
   sub-rewards. It is only necessary to determine the number of options.
 - **Learn termination with Advantage Function**: An interesting result of the paper is that the Advantage
   Function, used in many Actor-Critic RL algorithms, appears naturally in the derivation of the termination
   function gradient. This makes the gradient update go for an early termination when the advantage is negative,
   and a continuation of the option while the advantage is high.
 - **Flexible architecture**: The Option-Critic Architecture is flexible and many learning techniques can be
   employed. The experiments in the paper show how to deal with tabular learning as well as neural network
   function approximations.
 - **Improved learning in similar tasks**: This paper shows that, compared to "flat" RL algorithms such as SARSA(0)
   and flat Actor-Critic, Option-Critic-based algorithms learn faster new tasks similar to those they were trained
   with.

## Weaknesses
 - **Non modularity**: It is often difficult, if not impossible, to understand what these options do. By not
   employing sub-goals or pseudo-rewards, the learned options are not restricted to learn and well-defined behavior.
   This makes impossible or tied to very specific situations reusing single options into similar tasks. Therefore,
   we cannot "take" an option and plug it into another agent. We cannot create a more complex task from options
   learned in smaller tasks. If we trained a robot to go around a warehouse, we have no guarantee that
   one of the options learned will be related, for example, to the task of opening a door. This will make it impossible
   to just pick that option and plug it into a similar robot that does a more complex task. We can generally say that,
   although the learned options show generalization capabilities, they are still tied to their particular configuration.
