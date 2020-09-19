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

**Remember:** Given a dataset $$\mathcal{D} = \{x^1, ... x^K \}$$ of K n-dimensional vectors, autoregressive generative models learn its underlying distribution $$p(x)$$ by making use of the chain rule of probability:

\begin{equation}
p(x) = \prod_i^n p(x_i \mid x_{< i})
\end{equation}

Each of these conditional probabilities $$p(x_i \mid x_{< i})$$ can be modelled by a single RNN.
We can use this tractable $$p(x)$$ for sample **density estimation**, **sample** new datapoints or **missing data completion**.
In particular, this paper is interested in natural image completion.

**Problem:** Sequentially learning $$p(x_i \mid x_{< i})$$ is **order-dependent**. While temporal and sequential data have natural orders, 2D data (such as images) doesn’t.
Previous work (e.g. [PixelRNN](https://arxiv.org/abs/1601.06759) or [PixelCNN](https://arxiv.org/abs/1606.05328)) only trains on [raster scan order](https://en.wikipedia.org/wiki/Raster_scan) (left to right, top to bottom).
But this is only 1 of the $$n!$$ possible image traversal ordering!
This means inference can only be "reliably" done in this order.
Therefore, it fails in image-completing tasks when a lot of data is missing around first rows of the image:

{% include figure.html url="/_papers/LMConv/order.png" description="Figure 1: PixelCNN++ failing at image completion task because it cannot take advantage of the information in last rows of the image. LMConv can work in any order and uses all the available information to complete the image." zoom="1.0"%}

This work addresses this issue by adding 2 new ideas to [PixelCNN](https://arxiv.org/abs/1606.05328): Train the model on different **traversal order permutations**, and use **masking** on the level of features (unlike in the weights or inputs).

Training in arbitrary orders allows to customize the traversal for each task.
I.e. in an image completion traverse first the area with more information and obtain better conditional approximations.

### Pixel traversal permutation training



### Local masking

Bare with me because things get a bit convoluted at this point (pun intended).

## Results

## Contribution

- Extension of [PixelCNN](https://arxiv.org/abs/1606.05328) to estimate more reliable likelihoods in arbitrary orders.

## Weaknesses

- Transfer learning?