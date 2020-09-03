---
layout: paper
title: A Simple Baseline for Bayesian Uncertainty in Deep Learning
category: other
permalink: /papers/maddox_et_al_bayesian_uncertainty_deep_learning
paper-author: W. J. Maddox, T. Garipov, P. Izmailov, D. Vetrov, A. G.Wilson
post-author: Federico Taschin
paper-year: 2019
paper-link: https://arxiv.org/abs/1902.02476
---
<!--
Disclaimer and authorship:
This article is provided for free only for your personal informational and entertainment purposes. No commercial use of it is allowed.

Please note there might be mistakes. We would be grateful to receive (constructive) criticism if you spot any. You can reach us at: ai.campus.ai@gmail.com or directly open an issue on our github repo: https://github.com/CampusAI/CampusAI.github.io

If considering to use the text please cite the original author/s of the lecture/paper.
Furthermore, please acknowledge our work by adding a link to our website: https://campusai.github.io/ and citing our names: Oleguer Canal and Federico Taschin.
-->
This paper proposes the SWA-Gaussian algorithm for uncertainty representation and calibration in
Deep Learning. **CONTINUE ABSTRACT**


## The SWA algorithm
In their previous paper,
[Averaging Weights Leads to Wider Optima and Better Generalization, Feb 2019](https://arxiv.org/abs/1803.05407)
the authors proposed the SWA -Stochastic Weight Averaging- procedure for
training Deep Neural Networks, that leads to better generalization than standard training
methods. In SWA, training is performed with the common Stochastic Gradient Descent technique,
but in the final phase, an high or cyclical learning rate is used. After each epoch or cycle,
the resulting model is kept, and the final model is given by the average of a window of the
last models. 

As conjectured by [Chaudhari et al., 2017](https://arxiv.org/abs/1611.01838) and
[Keskar et al., 2017](https://arxiv.org/abs/1609.04836), local minima that are located in wide
valleys of the loss function are better for generalization. However, SGD solutions often lie on
the boundaries of these regions. This can be clearly seen in Figure 1: the SGD trajectory
explores points on the boundaries of the optimal with respect to the test error. It becomes then
desirable to average these points.

{% include figure.html url="/_papers/simple_baseline_uncertainty_dl/swa_error.png"
description="The L2-regularized cross-entropy train loss and test error surfaces of a
Preactivation ResNet-164 on CIFAR100 in the plane containing the first, middle and last points
(indicated by black crosses) in the trajectories with (left two) cyclical and (right two)
constant learning rate schedules. Picture from the SWA paper."
zoom=2 %}

## Bayesian methods
Bayesian methods aim to overcome the limitation of a Maximum Likelihood solution for the Neural
Network weights, which is often given by
