---
layout: article
title: "Parametric Deep Generative Models"
permalink: /lectures/generative_models
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

**Notation**: I will refer to datapoints as $$x$$ (usually high-dimensional), labels as $$y$$ and latent variables as $$z$$. Notice the similarity between $$y$$ and $$z$$, the only difference being $$y$$ is provided and $$z$$ are found by the model.

<!-- **NB**: _In this post we will focus on **parametric** distribution approximations.
In contrast to non-parametric ones, they
<span style="color:green">scale more efficiently with large datasets</span>
but
<span style="color:red">are limited in the family of distributions they can represent</span>._ -->

## Basics

Generative models fall in the realm of **unsupervised learning**:
A branch of machine learning which learns patterns from unlabeled data.
Most common unsupervised learning algorithms concern: [clustering](https://en.wikipedia.org/wiki/Cluster_analysis), [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction), or [latent variable models](https://en.wikipedia.org/wiki/Latent_variable_model).

Before venturing in more complex details of generative models, lets take a moment to refresh Bayes formula:

\begin{equation}
P(y | x) = \frac{P(x | y) P(y)}{P(x)}
\end{equation}

Naming goes: **Posterior**:  $$P(y \mid x)$$. **Likelihood**:  $$P(x \mid y)$$. **Prior**: $$P(y)$$. **Evidence**: $$P(x)$$.

## Discriminative vs Generative models

- **Discriminative models** task is to predict a label $$y$$ for any given datapoint $$x$$. I.e. learn the conditional probability distribution $$P(y \mid x)$$ (**posterior**) by mapping inputs to provided labels. (**supervised learning**)

- **Generative models** attempt to learn an **approximate probabilistic distribution** of $$P(x)$$, $$P(x \mid z)$$, or $$P(x \mid z)$$. Usually some functional form of $$P(z)$$ and $$P(X \mid z)$$ is assumed, then their parameters are estimated using data. If interested in the posterior one can use Bayes to compute it. (**unsupervised learning**)

Discriminative usually outperform generative models in classification tasks:

{% include figure.html url="/_lectures/ml/generative_models/dis_vs_gen.png" description="Figure 1: Learning a decision boundary $P(y \mid x)$ is easier than learning the full x distribution of each class $P(x \mid y)$ (Image from KTH DD2412 course)" zoom="1.0"%}

Nevertheless the rich interpretation generative models do of our data can be very useful. The next section presents some of their use-cases

## Generative models use-cases

So, imagine we have a dataset $$\mathcal{D}$$ of dog images and an algorithm capable of modelling its underlying distribution: $$P(X)$$.
We could:

- **Sample new datapoints** from $$P(X)$$ distribution. For instance, we could obtain new dog images beyond the observed ones by sampling from our modelled "dog image distribution".

- **Evaluate the probability of a sample $$x$$** by $$P(x)$$ (density estimation). We could use this to check how likely it is that a given image comes from the "dog image distribution" we used for training. Can be useful in uncertainty estimation to detect **out-of-distribution** (OOD) samples.

- **Infer latent variables** $$z$$. In the dog example we could understand the underlying common structure of dog images. These latent variables could be dog position, fur color, ears type...

**NB**: _Quantitative evaluation of generative models is non-trivial and is still being researched on. A common evaluation metric of $$P_\theta(x)$$ is to assess the negative log-likelihood (NNL) of a "test" set. Images from a dataset should have very high likelihood (they are samples of the distribution)._

**NB**: _Not all type of generative models are able to perform all of the above use-cases. There exist many different approaches (types) with its strengths and weaknesses._

# Generative models types

## Likelihood-based methods

They try to optimize the likelihood of the observed data for each data-point:

\begin{equation}
\mathcal{L} (x_i) = - \log p_\theta (x_i)
\end{equation}

Where:

\begin{equation}
p_\theta (x_i) = \int_z p_\theta (x_i \mid z) p(z) dz
\end{equation}

This means, they try to find the parametrized distribution $$p_\theta$$ which better explains the data.
Depending on how they fit this distributions we can divide them into: **Autoregressive models** (AR), **Variational autoencoders** (VAEs), and **Flow-based generative models**


### Autoregressive models (AR)

Given a dataset $$\mathcal{D} = \{x^1, ... x^K \}$$ of K n-dimensional datapoints $$x$$ ($$x$$ could be a flattened image for instance) we can apply the chain rule of probability to each dimension of the datapoint (we take the density estimation perspective):

\begin{equation}
p(x) = \prod_i^n p(x_i \mid x_{< i})
\end{equation}

**NB**: _This decomposition converts the joint modelling problem $$p(x_1, ..., x_n)$$ into a sequence modeling one._

**NB**: _A Bayesian network which does not do any assumption on the conditional independence of the variables is set to obey the **autoregressive property**._

Autoregressive models fix an ordering of the variables and model each conditional probability $$p(x_i \mid x_{< i})$$.
This model is composed by a parametrized function with a fixed number of params.
In practice fitting each of the distributions is computationally infeasible (too many parameters for high-dimensional inputs).

Simplification methods:

- **Independence assumption**: Instead of each variable dependent on all the previous, you could define a probabilistic graphical model and define some dependencies: $$P(x) \simeq \prod_i^n p \left(x_i \mid \{ x_j \}_{j \in parents_i} \right)$$. For instance, one could do Markov assumptions: $$P(x) \simeq \prod_i^n p \left(x_i \mid x_{i-1} \right)$$. More on this [paper](http://www.iro.umontreal.ca/~lisa/pointeurs/bb_2000_nips.pdf) and this other [paper](https://papers.nips.cc/paper/1153-does-the-wake-sleep-algorithm-produce-good-density-estimators.pdf).

- **Parameter reduction**: To ease the training one can under-parametrize the model and apply VI to find the closest distribution in the working sub-space. For instance you could design the conditional approximators parameters to grow linearly in input size like: $$P(x) \simeq \prod_i^n p \left(x_i \mid x_{< i}, \theta_i \right)$$ where $$\theta_i \in \mathcal{R}^i$$. More of this [here](https://www.sciencedirect.com/science/article/pii/0004370292900656).

- **Increase representation power**: I.e. parametrize $$p(x_i \mid x_{< i})$$ with an ANN. Parameters can either remain constant or increase with $$i$$ (see figure 2). In addition you can make these networks **share parameters** to ease the learning.

{% include figure.html url="/_lectures/ml/generative_models/ar_ann.png" description="Figure 2: Growing ANN modelling of the conditional distributions. (Image from KTH DD2412 course" zoom="1.0"%}

**NB**: _The order in which you traverse the data matters! While temporal and sequential data have natural orders, 2D data doesn't. A solution is to train an ensemble with different orders (ENADE) and average its predictions._

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

### Normalizing flow models

The main idea is to learn a deterministic [bijective](https://en.wikipedia.org/wiki/Bijection) (invertible) **mapping** from **easy distributions** (easy to sample and easy to evaluate density, e.g. Gaussian) to the **given data distribution** (more complex).

First we need to understand the **change of variables formula**: Given $$Z$$ and $$X$$ random variables related by a bijective (invertable) mapping $$f : \mathbb{R}^n \rightarrow \mathbb{R}^n$$ such that $$X = f(Z)$$ and $$Z = f^{-1}(X)$$ then:

\begin{equation}
p_X(x) = p_Z \left( f^{-1} (x) \right) \left|\det \left( \frac{\partial f^{-1} (x)}{\partial x} \right)\right|
\end{equation}

Were $$\frac{\partial f^{-1} (x)}{\partial x}$$ is the $$n \times n$$ Jacobian matrix of $$f^{-1}$$.
Notice that its determinant models the **local** change of volume of $$f^{-1}$$ at the evaluated point.

**NB:** _"**Normalizing**" because the change of variables gives a normalized density after applying the transformations (achieved by multiplying with the Jacobian determinant). "**Flow**" because the invertible transformations can be composed with each other to create more complex invertible transformations: $$f = f_0 \circ ... \circ f_k$$._

{% include figure.html url="/_lectures/ml/generative_models/normalizing-flow.png" description="Figure 3: Normalizing flow steps example from 1D Gaussian to a more complex distribution. (Image from lilianweng.github.io" zoom="1.0"%}

As you might have guessed, normalizing flow models parametrize this $$f$$ mapping function using an ANN $$(f_\theta)$$.
**This ANN**, however, needs to verify some specific architectural structures:

- <span style="color:red">Needs to be **deterministic**</span>
- <span style="color:red">I/O **dimensions** must be the **same** ($$f$$ has to be bijective)</span>
- <span style="color:red">Transformations must be **invertible**</span>
- <span style="color:red">Computation of the determinant of the Jacobian must be **efficient** and **differentiable**.</span>

Nevertheless they solve both previous approach problems:
- <span style="color:green">Present feature learning</span>.
- <span style="color:green">Present a tractable marginal likelihood</span>.

Most famous normalizing flow architectures ([NICE](https://arxiv.org/abs/1410.8516), [RealNVP](https://arxiv.org/abs/1605.08803), [Glow](https://arxiv.org/abs/1807.03039)) design layers whose Jacobian matrices are triangular or can be decomposed in triangular shape. These layers include variations of the **affine coupling layer**, **activation normalization layer** or **invertible 1x1 conv**.
Check out our [Glow paper post](/papers/glow) for further details on these layers.

**NB**: _Some models combine the flows with the autoregressive idea creating **autoregressive flows**: Each dimension in the input is conditioned to all previous ones. Check out [MAF](https://arxiv.org/abs/1705.07057) and [IAF](https://arxiv.org/abs/1606.04934)._

**NB**: _Similarly, flows can be applied to make VAEs latent space distribution more complex than Gaussian. Check out [f-VAES](rxiv.org/abs/1809.05861)._

<!-- ## Likelihood free learning

These models are not trained using maximum likelihood.

### Generative Adversarial Networks (GANs) -->
**_[Stay tunned for VAES and GANs explanations]_**