---
layout: page
title: Machine Learning
permalink: /ml/
---

<!-- This page contains explanations of diverse ML topics we found interesting. -->

<!-- ## Unsupervised Learning -->

## Generative Models

{% include paper-card.html
title="Why generative models?"
subtitle="Basics, Discriminative vs Generative, Use-cases, Types"
url="/ml/generative_models"%}

{% include paper-card.html
title="From Expectation Maximization to Variational Inference"
subtitle="Latent Variable Models, EM, VI, Amortized VI, Reparametrization Trick, Mean Field VI"
url="/ml/variational_inference"%}

{% include paper-card.html
title="Autoregressive models (AR)"
subtitle="Basics, Simplification methods, Pro/Cons, Relevant Papers"
url="/ml/autoregressive_models"%}

{% include paper-card.html
title="Normalizing flows"
subtitle="Basics, Pro/Cons, Relevant Papers"
url="/ml/flow_models"%}

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


## Annex

{% include paper-card.html
title="Probability Basics"
subtitle="Information theory, Statistical distances"
url="/ml/prob_modelling"%}

{% include paper-card.html title="Variational Inference Annex" subtitle="" url="/lectures/variational_inference_annex"   %}
