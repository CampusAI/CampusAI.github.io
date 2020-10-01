---
layout: article
title: "Autoregressive models (AR)"
permalink: /ml/autoregressive_models
content-origin: Standford CS236, KTH DD2412, lilianweng.github.io
post-author: Oleguer Canal
---
<!--
Disclaimer and authorship:
This article is provided for free only for your personal informational and entertainment purposes. No commercial use of it is allowed.

Please note there might be mistakes. We would be grateful to receive (constructive) criticism if you spot any. You can reach us at: ai.campus.ai@gmail.com or directly open an issue on our github repo: https://github.com/CampusAI/CampusAI.github.io

If considering to use the text please cite the original author/s of the lecture/paper.
Furthermore, please acknowledge our work by adding a link to our website: https://campusai.github.io/ and citing our names: Oleguer Canal and Federico Taschin.
-->

{% include start-row.html %}

Given a dataset $$\mathcal{D} = \{x^1, ... x^K \}$$ of K n-dimensional datapoints $$x$$ ($$x$$ could be a flattened image for instance) we can apply the chain rule of probability to each dimension of the datapoint (we take the density estimation perspective):

{% include end-row.html %}
{% include start-row.html %}

\begin{equation}
p(x) = \prod_i^n p(x_i \mid x_{< i})
\end{equation}

{% include annotation.html %}

This decomposition converts the joint modelling problem $$p(x_1, ..., x_n)$$ into a sequence modeling one.

A Bayesian network which does not do any assumption on the conditional independence of the variables is set to obey the **autoregressive property**.

{% include end-row.html %}
{% include start-row.html %}

Autoregressive models fix an ordering of the variables and model each conditional probability $$p(x_i \mid x_{< i})$$.
This model is composed by a parametrized function with a fixed number of params.
In practice fitting each of the distributions is computationally infeasible (too many parameters for high-dimensional inputs).

Simplification methods:

- **Independence assumption**: Instead of each variable dependent on all the previous, you could define a probabilistic graphical model and define some dependencies: $$P(x) \simeq \prod_i^n p \left(x_i \mid \{ x_j \}_{j \in parents_i} \right)$$. For instance, one could do Markov assumptions: $$P(x) \simeq \prod_i^n p \left(x_i \mid x_{i-1} \right)$$. More on this [paper](http://www.iro.umontreal.ca/~lisa/pointeurs/bb_2000_nips.pdf) and this other [paper](https://papers.nips.cc/paper/1153-does-the-wake-sleep-algorithm-produce-good-density-estimators.pdf).

- **Parameter reduction**: To ease the training one can under-parametrize the model and apply VI to find the closest distribution in the working sub-space. For instance you could design the conditional approximators parameters to grow linearly in input size like: $$P(x) \simeq \prod_i^n p \left(x_i \mid x_{< i}, \theta_i \right)$$ where $$\theta_i \in \mathcal{R}^i$$. More of this [here](https://www.sciencedirect.com/science/article/pii/0004370292900656).

- **Increase representation power**: I.e. parametrize $$p(x_i \mid x_{< i})$$ with an ANN. Parameters can either remain constant or increase with $$i$$ (see figure 2). In addition you can make these networks **share parameters** to ease the learning.

{% include end-row.html %}
{% include start-row.html %}

{% include figure.html url="/_ml/generative_models/ar_ann.png" description="Figure 2: Growing ANN modelling of the conditional distributions. (Image from KTH DD2412 course" zoom="1.0"%}

{% include annotation.html %}

The order in which you traverse the data matters! While temporal and sequential data have natural orders, 2D data doesn't. A solution is to train an ensemble with different orders (ENADE) and average its predictions.

{% include end-row.html %}
{% include start-row.html %}

Instead of having a static model for each input, we can use a **RNN** and encode the seen "context" information as hidden inputs. They work for sequences of arbitrary lengths and we can tune their modeling capacity. The only downsides are that they are slow to train (sequential) and might present vanishing/exploding gradient problems.

[PixelRNN](https://arxiv.org/abs/1601.06759) applies this idea to images.
They present some tricks like multi-scale context to achieve better results than just traversing the pixels row-wise. It consists of first traversing sub-scaled versions of the image to finally fit the model on the whole image.
If interested, check out our [LMConv post](/papers/LMConv).
Some other interesting papers about this topic: [PixelCNN](https://arxiv.org/abs/1606.05328) [WaveNet](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)

Overall AR provide:

- <span style="color:green">**Tractable likelihoods**: exact and simple density estimation)</span>
- <span style="color:green">**Simple generation process**, which is very good for data imputation (specially if available data is at the beginning of the input sequence)</span>

But:

-  <span style="color:red">There is no direct mechanism for learning features (**no encoding**)</span>.
-  <span style="color:red">**Slow**: training, sample generation, and density estimation. Because of the sequential nature of the algorithm</span>.


<!-- ### Variational autoencoders (VAE)

TODO: Explanation
$$p_\theta(x) = \int p_\theta(x, z) dz = \int p_\theta(x \mid z) p(z) dz$$.
Where $$p_\theta(x \mid z)$$ is modelled by the decoder network and $$p(z)$$ the chosen prior for the latent variables $$z$$.

Latent coding perspective

Use of Variational INference approximation trick to avoid the integration.

Can <span style="color:green">learn feature representations</span> $$(z)$$ but <span style="color:red">have intractable marginal likelihood</span> $$p_\theta(x \mid z)$$. -->

{% include end-row.html %}