---
layout: page
title: Research
permalink: /experiments/
index: true
---

## Own Experiments

{% include card.html
title="Self-learned vehicle control using PPO"
img="/_experiments/autonomous_driving/icon.gif"
url="/experiments/autonomous_driving"
brief="This work tackles the completion of an obstacle maze by a self-driving vehicle in a realistic physics environment.
We tested the adaptability of our algorithm by learning both the controls of a car and a drone.
The agents are trained using the PPO (policy gradient) algorithm and using curriculum learning for faster training.;
This project was part of a contest and we achieved the faster maze completion times against heuristic approaches.
"
type="description" %}

{% include card.html
title="NN Surgery in Deep RL"
img="/_experiments/nn_surgery/lunar_lander.gif"
url="/experiments/nn_surgery"
brief="In this work we experiment on weight transplantation after slight ANN modifications.
Networks can be modified by increasing or reducing the number of input/outputs as well as the number of hidden layers and units.
We show that when modifying the network structure, weight transplant achieves faster and better results than training from scratch."
type="description" %}

Stay tunned for new experiments such as a RL-trained [**RocketLeague**](https://en.wikipedia.org/wiki/Rocket_League) bot and a **Terrarium** where animals were trained to survive while following a simple food chain.

<!--
To add an experiment one must add a line with the following code:

include media_card.html title="" brief="" img="" url="" type=""

title:  The title of the lecture 
brief:  A string of ; separated sentences that will be put in a bullet list
img:    An image that represents the lecture
url:    The url of the lecture post
type:   The type of the card. Here we use "description".
-->

<br>
## Paper reviews

### Uncertainty Estimation

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
star="yes"
%}

<br>

### ANN Explainability

{% include paper-card.html title="Understanding Black-box Predictions via Influence Functions"
url="/papers/Understanding-Black-box-Predictions-via-Infuence-Functions"
subtitle="Pang Wei Koh, Percy Liang, 2017"
%}

{% include paper-card.html title="Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
url="/papers/Grad-CAM"
subtitle=" Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra, 2019"
%}

<br>

### Generative Models

{% include paper-card.html title="Glow: Generative Flow with Invertible 1x1 Convolutions"
url="/papers/glow"
subtitle="Diederik P. Kingma, Prafulla Dhariwal, 2018"
%}

{% include paper-card.html title="LMConv: Locally Masked Convolutions for Autoregressive Models"
url="/papers/LMConv"
subtitle="Ajay Jain, Pieter Abbeel, Deepak Pathak, 2020"
%}

<br>

### Beyond standard labels

<!-- include paper-card.html title="A Simple Framework for Contrastive Learning of Visual Representations"
subtitle="Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton"
url="chen2020-contrastive-learning"
-->

{% include paper-card.html
title="Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels" 
subtitle="Zhilu Zhang, Mert R. Sabuncu"
url="/papers/zhang_sabuncu2018_cross_entropy_noisy_labels"
%}

<br>
### Reinforcement Learning
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

{% include paper-card.html title="Neural Network Surgery with Sets"
url="/papers/NN_surgery_sets" subtitle="Jonathan Raiman, Susan Zhang, Christy Dennison, 2019"   %}
