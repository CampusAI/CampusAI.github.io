---
layout: paper
title: "Glow: Generative Flow with Invertible 1x1 Convolutions"
category: Generative models
permalink: /papers/glow
paper-author: Diederik P. Kingma, Prafulla Dhariwal
post-author: Oleguer Canal
paper-year: 2018
paper-link: https://arxiv.org/abs/1807.03039
---
<!--
Disclaimer and authorship:
This article is provided for free only for your personal informational and entertainment purposes. No commercial use of it is allowed.

Please note there might be mistakes. We would be grateful to receive (constructive) criticism if you spot any. You can reach us at: ai.campus.ai@gmail.com or directly open an issue on our github repo: https://github.com/CampusAI/CampusAI.github.io

If considering to use the text please cite the original author/s of the lecture/paper.
Furthermore, please acknowledge our work by adding a link to our website: https://campusai.github.io/ and citing our names: Oleguer Canal and Federico Taschin.
-->

If not familiar with flow-based generative models I suggest to first take a look at our [Parametric Deep Generative Models post](http://127.0.0.1:4000/lectures/generative_models).

## Idea

This paper presents a flow-based deep generative model extending from [NICE](https://arxiv.org/abs/1410.8516), [RealNVP](https://arxiv.org/abs/1605.08803) algorithms.
Its main novelty its the design of a distinct flow implementing a new **invertible 1x1 conv layer**.

The proposed flow is composed by a multi-scale architecture (same as in [RealNVP](https://arxiv.org/abs/1605.08803) where each step has 3 distinct flow-layers combined:

{% include figure.html url="/_papers/Glow/architecture.png" description="Figure 1: Each step of the proposed flow consists of three layers: an actnorm, the new invertible 1x1 conv and an affine coupling layer" zoom="1.0"%}

#### Actnorm: Scale and bias layer

#### Invertible 1x1 convolution

#### Affine coupling layers

In summary this is each layer function, its inverse and its Jacobian log-determinant:

{% include figure.html url="/_papers/Glow/layers.png" description="Figure 2: Layer summary" zoom="1.0"%}

## Results

### Quantitative

### Qualitative


## Contribution

- Design of a new normalizing-flow layer: The **invertible 1x1 convolution**.

- Improved quantitative SOTA results (in terms of log-likelihood of the modelled distribution $$P(X)$$).

## Weaknesses
