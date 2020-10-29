---
layout: page
title: Papers
permalink: /papers/
---

This page contains summaries and annotations from papers we found interesting.


## Uncertainty Estimation

{% include paper-card.html
title="Weight Uncertainty in Neural Networks"
url="/papers/weight-uncertainty"
subtitle="Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, Daan Wierstra, 2015"  %}

{% include paper-card.html
title="Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"
url="/papers/uncertainty-estimation-deep-ensembles"
subtitle="Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell, 2017"  %}

{% include paper-card.html
title="Out-of-Distribution Detection Using an Ensemble of Self Supervised Leave-out Classifiers"
url="/papers/out_of_distribution_detection_ensemble"
subtitle="A. Vyas, N. Jammalamadaka, X. Zhu, D. Das, B. Kaul, T. L. Willke, 2018"
%}

{% include paper-card.html title="A Simple Baseline for Bayesian Uncertainty in Deep Learning"
url="/papers/maddox_et_al_bayesian_uncertainty_deep_learning"
subtitle="Wesley J. Maddox, Timur Garipov, Pavel Izmailov, Dmitry Vetrov, Andrew Gordon Wilson, 2019"
%}

{% include paper-card.html title="Ensemble Distribution Distillation"
url="/papers/ensemble-distribution-distillation"
subtitle="Andrey Malinin, Bruno Mlodozeniec, Mark Gales, 2019"
%}


## ANN Explainability

{% include paper-card.html title="Understanding Black-box Predictions via Influence Functions"
url="/papers/Understanding-Black-box-Predictions-via-Infuence-Functions"
subtitle="Pang Wei Koh, Percy Liang, 2017"
%}

{% include paper-card.html title="Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
url="/papers/Grad-CAM"
subtitle=" Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra, 2019"
%}


## Generative Models

{% include paper-card.html title="Glow: Generative Flow with Invertible 1x1 Convolutions"
url="/papers/glow"
subtitle="Diederik P. Kingma, Prafulla Dhariwal, 2018"
%}

{% include paper-card.html title="LMConv: Locally Masked Convolutions for Autoregressive Models"
url="/papers/LMConv"
subtitle="Ajay Jain, Pieter Abbeel, Deepak Pathak, 2020"
%}


## Beyond standard labels

<!-- include paper-card.html title="A Simple Framework for Contrastive Learning of Visual Representations"
subtitle="Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton"
url="chen2020-contrastive-learning"
-->

{% include paper-card.html
title="Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels" 
subtitle="Zhilu Zhang, Mert R. Sabuncu"
url="/papers/zhang_sabuncu2018_cross_entropy_noisy_labels"
%}


## RL Algorithms

{% include paper-card.html title="Soft Actor-Critic (SAC)"
subtitle="T. Haarnoja, A. Zhou, P. Abbeel, S. Levine, 2018" url="/papers/Soft-Actor-Critic"   %}
<!-- This paper approaches the high sample complexity of on-policy RL and the brittle convergence of off-policy RL by introducing Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. -->

{% include paper-card.html title="Hierarchical Reinforcement Learning with MAXQ Value Function Decomposition"
url="/papers/HRL-with-MAXQ-decomposition"  subtitle="Thomas G. Dietterich, 2000"  %}
<!-- This paper presents MAXQ decomposition: a method to decompose the Value Function for a given hierarchical policy in a recursive fashion. -->

{% include paper-card.html title="The Option-Critic Architecture"
subtitle="Pierre-Luc Bacon, Jean Harb, Doina Precup, 2016"
url="/papers/the-option-critic-architecture" %}
<!-- The Options framework provides theoretical grounds for temporal abstraction in Reinforcement Learning. Each Option can be considered as a macro-action with its policy and termination condition, leading to two levels of policies: one policy over options and several intra-option policies. This paper presents the Option-Critic Architecture."
url="/papers/the-option-critic-architecture -->


## Other

{% include paper-card.html title="Neural Network Surgery with Sets"
url="/papers/NN_surgery_sets" subtitle="Jonathan Raiman, Susan Zhang, Christy Dennison, 2019"   %}
