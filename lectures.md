---
layout: page
title: Theory
permalink: /theory/
---

## Lecture notes from Sergey Levine UC Berkeley CS-285 (2019)

In this section you can find our summaries from [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/) (Google, UC Berkeley): UC Berkeley CS-285 [Deep Reinforcement Learning course](http://rail.eecs.berkeley.edu/deeprlcourse/).

<!--
To add a lecture one must add a line with the following code:

{% include media_card.html title="" brief="" img="" url="" type="" %}

title:  The title of the lecture 
brief:  A string of ; separated sentences that will be put in a bullet list
img:    An image that represents the lecture
url:    The url of the lecture post
type:   The type of the card. Here always use "bulletlist"  
-->

{% include media_card.html title="Lecture 1: Introduction" brief="SL vs RL;Ways to learn;How to build intelligent machines;State of the art" img="/_lectures/lecture_1/icon.jpg" url="/lectures/lecture1" type="bulletlist" %}


{% include media_card.html title="Lecture 2: Imitation Learning" brief="Behavioral Cloning;Why doesn't it work" img="/_lectures/lecture_2/icon.png" url="/lectures/lecture2" type="bulletlist" %}


{% include media_card.html title="Lecture 4: Introduction to Reinforcement Learning" brief="Markov Decision Process;The goal of RL;RL algorithms" img="/_lectures/lecture_4/icon.png" url="/lectures/lecture4" type="bulletlist" %}


{% include media_card.html title="Lecture 5: Policy Gradients" brief="Policy differentiation;The REINFORCE algorithm;Solving the causality issue;Baselines;Off-policy Policy Gradient" img="/_lectures/lecture_5/icon.png" url="/lectures/lecture5" type="bulletlist" %}


{% include media_card.html title="Lecture 6: Actor-Critic (AC) Algorithms" brief="Policy Gradients variance reduction;Policy Evaluation (Monte Carlo vs Bootstrapping);Infinite horizon problems;Batch AC algorithm;Online AC algorithm" img="/_lectures/lecture_6/icon.png" url="/lectures/lecture6" type="bulletlist" %}


{% include media_card.html title="Lecture 7: Value Function Methods" brief="Policy Iteration;Value Iteration;Q iteration with Deep Learning;Q Learning;Exploration" img="/_lectures/lecture_7/icon.png" url="/lectures/lecture7" type="bulletlist" %}


{% include media_card.html title="Lecture 8 Deep RL with Q-functions" brief="Replay buffer and target network;DQN (Deep Q Networks);Double Q-Learning;Multi-step returns;Continuous actions;DDPG (Deep Deterministic Policy Gradient)" img="/_lectures/lecture_8/icon.png" url="/lectures/lecture8" type="bulletlist" %}


{% include media_card.html title="Lecture 9: Advanced Policy Gradients" brief="Policy Gradient as Policy Iteration;The KL Divergence constraint;Dual Gradient Descent;Natural Gradients and Trust Region Policy Optimization;Proximal Policy Optimization" img="/_lectures/lecture_9/icon.png" url="/lectures/lecture9" type="bulletlist" %}

{% include media_card.html title="Lecture 10: Model-based Planning" brief="Deterministic vs Stochastic environments;Stochastic optimization methods;Monte Carlo Tree Search (MCTS);Collocation trajectory optimization;Shooting trajectory optimization" img="/_lectures/lecture_10/icon.png" url="/lectures/lecture10" type="bulletlist" %}

{% include media_card.html title="Lecture 11: Model-based Reinforcement Learning" brief="Naive Model-Based RL;Uncertainty in model-based RL;Model-based RL with complex observations" img="/_lectures/lecture_11/icon.png" url="/lectures/lecture11" type="bulletlist" %}

{% include media_card.html title="Lecture 12: Model-based Policy Learning" brief="How to use env. models to learn policies;Local vs Global policies;Guided policy search;Policy Distillation;Divide & conquer RL" img="/_lectures/lecture_12/icon.png" url="/lectures/lecture12" type="bulletlist" %}

{% include media_card.html title="Lecture 13: Variational Inference and Generative Models" brief="Latent Variable Models;Variational Inference;Amortized Variational Inference;Reparametrization Trick;" img="/_lectures/lecture_13/icon.png" url="/lectures/lecture13" type="bulletlist" %}

<!-- {% include media_card.html title="Lecture 14: Control as inference" brief="" img="" url="" %} -->

{% include media_card.html title="Lecture 15: Inverse Reinforcement Learning" brief="Underspecification problem;Feature Matching IRL;Maximum Entropy IRL" img="/_lectures/lecture_15/icon.png" url="/lectures/lecture15" type="bulletlist" %}

{% include media_card.html title="Lecture 16: Transfer and Multi-task Learning" brief="Forward Transfer;Multi-task Transfer" img="/_lectures/lecture_16/icon.png" url="/lectures/lecture16" type="bulletlist" %}
<!-- {% include media_card.html title="Lecture 17: Distributed RL" brief="" img="" url="" %} -->
<!-- {% include media_card.html title="Lecture 18: Exploration (Part 1)" brief="" img="" url="" %} -->
<!-- {% include media_card.html title="Lecture 19: Exploration (Part 2)" brief="" img="" url="" %} -->
<!-- {% include media_card.html title="Lecture 20: Meta-learning" brief="" img="" url="" %} -->
<!-- {% include media_card.html title="Lecture 21: Information Theory, Open Problems" brief="" img="" url="" %} -->

## Annex

This section contains both basic RL knowledge assumed to be known in the previous course and some demonstrations which we found interesting to add as an annex.
In addition we added our own interpretations of some concepts hoping they can ease their understanding.

- [Annex 1: MDP Basics](/lectures/basic_concepts)
  
- [Annex 2: Policy Expectations, Explained](/lectures/policy_expectations)

- [Annex 5: Policy Gradients](/lectures/policy_gradients_annex)

- [Annex 13: Variational Inference](/lectures/variational_inference_annex)

## Other great resources

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/bookdraft2017nov5.pdf), Sutton & Barto, 2017. (Arguably the most complete RL book out there)

- [David Silver](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Home.html) (DeepMind, UCL): UCL COMPM050 [Reinforcement Learning course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html).

- [Lil'Log](https://lilianweng.github.io/lil-log/) blog does and outstanding job at explaining algorithms and recent developments in both RL and SL.

- This RL [dictionary](https://towardsdatascience.com/the-complete-reinforcement-learning-dictionary-e16230b7d24e) can also be useful to keep track of all field-specific terms.

- If looking for some motivation to learn about DRL don't miss this truly inspiring [documentary](https://www.youtube.com/watch?v=WXuK6gekU1Y) on DeepMind's AlphaGo algorithm. 
