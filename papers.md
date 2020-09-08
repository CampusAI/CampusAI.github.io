---
layout: page
title: Papers
permalink: /papers/
---
<!--
To add a paper one must add a line with the following code:

{% include card.html title="" brief="" img="" url="" type="" %}

title:      The title of the paper
img:        An image that represents the paper, or leave "" for no image
url:        The url of the paper post
type:       "bulletlist" or "description".
brief:      The text content of the card. If type is "bulletlist",
            semicolons are used to split the text into bullet points.
            If type is "description", semicolons are parsed as newlines.
subtitle:   Put here the paper authors and year
-->

This page contains summaries and annotations from papers we found interesting. 
We mainly review RL-related papers but you'll also find more general ML topics.

## RL Algorithms

{% include card.html title="Soft Actor-Critic (SAC)"
brief=""
subtitle="T. Haarnoja, A. Zhou, P. Abbeel, S. Levine, 2018" url="/papers/Soft-Actor-Critic" img="" type="description" %}
<!-- This paper approaches the high sample complexity of on-policy RL and the brittle convergence of off-policy RL by introducing Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. -->

## Hierarchical RL
{% include card.html title="Hierarchical Reinforcement Learning with MAXQ Value Function Decomposition"
brief=""
url="/papers/HRL-with-MAXQ-decomposition" img="" subtitle="Thomas G. Dietterich, 2000" type="description" %}
<!-- This paper presents MAXQ decomposition: a method to decompose the Value Function for a given hierarchical policy in a recursive fashion. -->

{% include card.html title="The Option-Critic Architecture"
brief=""
subtitle="Pierre-Luc Bacon, Jean Harb, Doina Precup, 2016" img="" type="description" %}
<!-- The Options framework provides theoretical grounds for temporal abstraction in Reinforcement Learning. Each Option can be considered as a macro-action with its policy and termination condition, leading to two levels of policies: one policy over options and several intra-option policies. This paper presents the Option-Critic Architecture."
url="/papers/the-option-critic-architecture -->

## Uncertainty Estimation

{% include card.html title="A Simple Baseline for Bayesian Uncertainty in Deep Learning"
brief=""
url="/papers/maddox_et_al_bayesian_uncertainty_deep_learning"
subtitle="Wesley J. Maddox, Timur Garipov, Pavel Izmailov, Dmitry Vetrov, Andrew Gordon Wilson, 2019"
img="" type="description" %}

{% include card.html
title="Out-of-Distribution Detection Using an Ensemble of Self Supervised Leave-out Classifiers"
brief=""
url="/papers/out_of_distribution_detection_ensemble"
subtitle="A. Vyas, N. Jammalamadaka, X. Zhu, D. Das, B. Kaul, T. L. Willke, 2018"
img="" type="description" %}

{% include card.html
title="Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"
brief=""
url="/papers/uncertainty-estimation-deep-ensembles"
subtitle="Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell, 2017" type="description" %}

{% include card.html
title="Weight Uncertainty in Neural Networks"
brief=""
url="/papers/weight-uncertainty"
subtitle="Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, Daan Wierstra, 2015" type="description" %}

{% include card.html title="Ensemble Distribution Distillation"
brief=""
url="/papers/ensemble-distribution-distillation"
subtitle="Andrey Malinin, Bruno Mlodozeniec, Mark Gales, 2019"
img="" type="description" %}

## Other

{% include card.html title="Neural Network Surgery with Sets"
brief=""
url="/papers/NN_surgery_sets" subtitle="Jonathan Raiman, Susan Zhang, Christy Dennison, 2019" img="" type="description" %}
<!-- This paper presents an approach to continuously train a Deep RL policy model while performing architecture and environment modifications. -->
