---
layout: page
title: Papers
permalink: /papers/
---

## RL Algorithms

{% include card.html title="Soft Actor-Critic (SAC)"
brief="This paper approaches the high sample complexity of on-policy RL and the brittle convergence of off-policy RL by introducing Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor."
paper-author="T. Haarnoja, A. Zhou, P. Abbeel, S. Levine" paper-year="2018" url="/papers/Soft-Actor-Critic" img="" type="description" %}

## Hierarchical RL
{% include card.html title="Hierarchical Reinforcement Learning with MAXQ Value Function Decomposition"
brief="This paper presents MAXQ decomposition: a method to decompose the Value Function for a given hierarchical policy in a recursive fashion."
url="/papers/HRL-with-MAXQ-decomposition" img="/_papers/HRL_with_MAXQ_decomposition/taxi_navigation.png" paper-author="Thomas G. Dietterich" paper-year="2000" type="description" %}


{% include card.html title="The Option-Critic Architecture"
brief="The Options framework provides theoretical grounds for temporal abstraction in Reinforcement Learning. Each Option can be considered as a macro-action with its policy and termination condition, leading to two levels of policies: one policy over options and several intra-option policies. This paper presents the Option-Critic Architecture."
url="/papers/the-option-critic-architecture" paper-author="Pierre-Luc Bacon, Jean Harb, Doina Precup" paper-year="2016" img="/_papers/the_option_critic_architecture/algo1optioncritic.png" type="description" %}

## Other

{% include card.html title="Neural Network Surgery with Sets"
brief="This paper presents an approach to continuously train a Deep RL policy model while performing architecture and environment modifications."
url="/papers/NN_surgery_sets" paper-author="Jonathan Raiman, Susan Zhang, Christy Dennison" paper-year="2019" img="/_papers/NN_surgery_sets/propagation.png" type="description" %}
