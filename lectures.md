---
layout: page
title: Theory
permalink: /theory/
---

## Lecture notes from Sergey Levine UC Berkeley CS-285 (2019)

In this section you can find our summaries from [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/) (Google, UC Berkeley): UC Berkeley CS-285 [Deep Reinforcement Learning course](http://rail.eecs.berkeley.edu/deeprlcourse/).

- [Lecture 1: Introduction](/lectures/lecture1)
    - Supervised Learning vs RL
    - Ways to learn
    - How to build intelligent machines
    - State of the art
- [Lecture 2: Imitation Learning](/lectures/lecture2)
    - Behavioral Cloning
    - Why doesn't it work?
<!-- - Lecture 3: TensorFlow Review -->
- [Lecture 4: Introduction to Reinforcement Learning](/lectures/lecture4)
    - Markov Decision Processes
    - The goal of RL
    - RL algorithms
- [Lecture 5: Policy Gradients](/lectures/lecture5)
    - Policy differentiation
    - The REINFORCE algorithm
    - Solving the causality issue
    - Baselines
    - Off-policy Policy Gradient
- [Lecture 6: Actor-Critic (AC) Algorithms](/lectures/lecture6)
    - Reducing Variance on Policy Gradients
    - Policy evaluation (MC vs Bootstrapping)
    - AC algorithm
- [Lecture 7: Value Function Methods](/lectures/lecture7)
    - Policy Iteration
    - Value Iteration
    - Q iteration with Deep Learning
    - Q Learning
    - Exploration
- [Lecture 8: Deep RL with Q-functions](/lectures/lecture8)
    - Replay buffer and target network
    - DQN (Double Q Networks)
    - Double Q-Learning
    - Multi-step returns
    - Continuous actions
    - DDPG (Deep Deterministic Policy Gradient)
- [Lecture 9: Advanced Policy Gradients](/lectures/lecture9)
    - Policy Gradient as Policy Iteration
    - The KL Divergence constraint
    - Dual Gradient Descent
    - Natural Gradients and Trust Region Policy Optimization
    - Proximal Policy Optimization
- [Lecture 10: Model-based Planning](/lectures/lecture10)
    - Stochastic optimization methods
    - Monte Carlo tree search (MCTS)
    - Trajectory optimization 
- [Lecture 11: Model-based Reinforcement Learning](/lectures/lecture11)
    - Naive Model-Based RL
    - Uncertainty in model-based RL
    - Model-based RL with complex observations
- [Lecture 12: Model-based Policy Learning](/lectures/lecture12)
    - How to use env. models to learn policies
    - Local vs Global policies
    - Guided policy search, policy distillation, divide & conquer RL
- Lecture 13: Variational Inference and Generative Models
- Lecture 14: Control as Inference
- [Lecture 15: Inverse Reinforcement Learning](/lectures/lecture15)
    - Feature matching IRL
    - Maximum Entropy IRL
- [Lecture 16: Transfer and Multi-task Learning](/lectures/lecture16)
    - Forward Transfer
    - Multi-task Transfer
- Lecture 17: Distributed RL
- Lecture 18: Exploration (Part 1)
- Lecture 19: Exploration (Part 2)
- Lecture 20: Meta-learning
- Lecture 21: Information Theory, Open Problems

## Annex

This section contains both basic RL knowledge assumed to be known in the previous course and some demonstrations which we found interesting to add as an annex.
In addition we added our own interpretations of some concepts hoping they can ease their understanding.

- [Annex 1: MDP Basics](/lectures/basic_concepts)
  
- [Annex 2: Policy Expectations, Explained](/lectures/policy_expectations)

- [Annex 5: Policy Gradients](/lectures/policy_gradients_annex)


## Other great resources

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/bookdraft2017nov5.pdf), Sutton & Barto, 2017. (Arguably the most complete RL book out there)

- [David Silver](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Home.html) (DeepMind, UCL): UCL COMPM050 [Reinforcement Learning course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html).

- [Lil'Log](https://lilianweng.github.io/lil-log/) blog does and outstanding job at explaining algorithms and recent developments in both RL and SL.

- This RL [dictionary](https://towardsdatascience.com/the-complete-reinforcement-learning-dictionary-e16230b7d24e) can also be useful to keep track of all field-specific terms.

- If looking for some motivation to learn about DRL don't miss this truly inspiring [documentary](https://www.youtube.com/watch?v=WXuK6gekU1Y) on DeepMind's AlphaGo algorithm. 