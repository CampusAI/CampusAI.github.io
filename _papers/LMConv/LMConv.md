---
layout: paper
title: "LMConv: Locally Masked Convolutions for Autoregressive Models"
category: Generative models
permalink: /papers/LMConv
paper-author: Ajay Jain, Pieter Abbeel, Deepak Pathak
post-author: Oleguer Canal
paper-year: 2020
paper-link: https://arxiv.org/abs/2006.12486
---
<!--
Disclaimer and authorship:
This article is provided for free only for your personal informational and entertainment purposes. No commercial use of it is allowed.

Please note there might be mistakes. We would be grateful to receive (constructive) criticism if you spot any. You can reach us at: ai.campus.ai@gmail.com or directly open an issue on our github repo: https://github.com/CampusAI/CampusAI.github.io

If considering to use the text please cite the original author/s of the lecture/paper.
Furthermore, please acknowledge our work by adding a link to our website: https://campusai.github.io/ and citing our names: Oleguer Canal and Federico Taschin.
-->

If not familiar with **autoregressive generative models** I suggest to first take a look at our [Parametric Deep Generative Models post](http://127.0.0.1:4000/lectures/generative_models).


## Idea

**Remember:** Given a dataset $$\mathcal{D} = \{x^1, ... x^K \}$$ of K n-dimensional vectors, autoregressive generative models ease the learning of its underlying distribution $$p(x)$$ by making use of the chain rule of probability:

\begin{equation}
p(x) = \prod_i^n p(x_i \mid x_{< i})
\end{equation}

Each of these conditional probabilities $$p(x_i \mid x_{< i})$$ can be modelled by a shared RNN.

**Problem:** Sequentially learning $$p(x_i \mid x_{< i})$$ is **order-dependent**. While temporal and sequential data have natural orders, 2D data (such as images) doesn’t.

Previous work (e.g. [PixelCNN](https://arxiv.org/abs/1606.05328)) only trains its 

This work addresses this issue while developing on [PixelCNN](https://arxiv.org/abs/1606.05328).
It mainly combines 2 ideas: Train the model on different permutations



### Pixel traversal permutation training

### Local masking

Bare with me because things get a bit convoluted at this point (pun intended).

## Results

## Contribution

## Weaknesses

- Transfer learning?