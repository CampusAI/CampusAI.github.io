---
layout: paper
title: Understanding Black-box Predictions via Influence Functions
category: Explainability
permalink: /papers/Understanding-Black-box-Predictions-via-Infuence-Functions
paper-author: Pang Wei Koh, Percy Liang
post-author: Oleguer Canal
paper-year: 2017
paper-link: https://arxiv.org/abs/1612.01474
---
<!--
Disclaimer and authorship:
This article is provided for free only for your personal informational and entertainment purposes. No commercial use of it is allowed.

Please note there might be mistakes. We would be grateful to receive (constructive) criticism if you spot any. You can reach us at: ai.campus.ai@gmail.com or directly open an issue on our github repo: https://github.com/CampusAI/CampusAI.github.io

If considering to use the text please cite the original author/s of the lecture/paper.
Furthermore, please acknowledge our work by adding a link to our website: https://campusai.github.io/ and citing our names: Oleguer Canal and Federico Taschin.
-->

## IDEA

ANNs parameters are given uniquely from the data.
But how do we know which data-points from our dataset are relevant?

[Influence functions](https://en.wikipedia.org/wiki/Robust_statistics#Empirical_influence_function) are a classic technique from [robust statistics](https://en.wikipedia.org/wiki/Robust_statistics) to identify the training points most responsible for a given prediction.
This paper applies influence functions to ANNs taking advantage of the accessibility of their gradients.

**Notation**:
- $$z_i = (x_i, y_i) \in \mathcal{D}$$ is a data-point of the dataset.
- $$L(z, \theta)$$ the model loss of that data-point for parameters $$\theta \in \mathbb{R}^p$$.
- $$\hat \theta := \arg \min_\theta \frac{1}{n} \sum_i^n L(z_i, \theta)$$ are the optimal parameters when training with the entire dataset (we assume all points have the same weight in the mean).

**Initial assumptions**: The loss function is twice-differentiable and convex in $$\theta$$.


### Upweighting a training point
The influence of upweighting $$z$$ on the parameters $$(I_{up, params}(z))$$
tells us how are the training parameters going to change if the weight of given point $$z$$ is increased by $$\epsilon$$. I.e $$I_{up, params}(z) := \frac{\partial \hat \theta_{\epsilon, z}}{\partial \epsilon} \vert_{\epsilon=0}$$.

Applying influence functions (and some Taylor-expansion approximations) we get:

\begin{equation}
I_{up, params}(z) = - H_{\hat \theta}^{-1} \cdot \nabla_\theta L(z, \hat \theta)
\end{equation}

Where $$H_{\hat \theta} \in \mathbb{R}^{p \times p}$$ is the Hessian of the loss function w.r.t $$\theta$$. It can be inverted since its positive definite (PD), thanks to the convexity assumption. $$\nabla_\theta L(z, \hat \theta) \in \mathbb{R}^{p \times 1}$$ is the gradient of the loss function w.r.t $$\theta$$ evaluated at $$z$$ with parameters $$\hat \theta$$.

**NB**: _If we take $$\epsilon = - \frac{1}{n}$$ we can see the effect on the parameters of removing a point $$z$$ from the dataset: $$\hat \theta_{-z} \simeq \hat \theta - \frac{1}{n} I_{up, params}(z)$$_.

$$I_{up, loss} (z, z_{test}) = \frac{\partial L(z_{test}, \hat \theta_{\epsilon, z})}{\partial \epsilon} \vert_{\epsilon=0}$$ then encodes "how important" a data-point $$z$$ is to a test-point $$z_{test}$$. Developing we get:

\begin{equation}
I_{up, loss} (z, z_{test}) = - \nabla_\theta L(z_{test}, \hat \theta)^T \cdot H_{\hat \theta}^{-1} \cdot \nabla_\theta L(z, \hat \theta)
\end{equation}

### Perturbing a training point
Using a similar reasoning, we can evaluate the influence a perturbation of some data-point $$z$$ given by $$\delta$$ can have on the loss of some test-point $$z_i$$:

\begin{equation}
I_{up, loss} (z, z_{test}) = - \nabla_\theta L(z_{test}, \hat \theta)^T \cdot H_{\hat \theta}^{-1} \cdot \nabla_x \nabla_\theta L(z, \hat \theta)
\end{equation}

### Assumptions and approximations