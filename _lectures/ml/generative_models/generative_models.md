---
layout: lecture
title: "Parametric Deep Generative Models"
permalink: /lectures/generative_models
lecture-author: Stefano Ermon, Aditya Grover
lecture-date: 2019
post-author: Oleguer Canal
slides-link: https://deepgenerativemodels.github.io/
---

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

{% include figure.html url="/_lectures/ml/generative_models/dis_vs_gen.png" description="Figure 1: Learning a decision boundary $P(y \mid x)$ is easier than learning the full x distribution of each class $P(x \mid y)$" zoom="1.0"%}

Nevertheless the rich interpretation generative models do of our data can be very useful. The next section presents some of their use-cases

## Generative models use-cases

So, imagine we have a dataset $$\mathcal{D}$$ of dog images and an algorithm capable of modelling its underlying distribution: $$P(X)$$.
We could:

- **Sample new datapoints** from $$P(X)$$ distribution. For instance, we could obtain new dog images beyond the observed ones by sampling from our modelled "dog image distribution".

- **Evaluate the probability of a sample $$x$$** by $$P(x)$$. We could use this to check how likely it is that a given image comes from the "dog image distribution" we used for training. Can be useful in uncertainty estimation to detect **out-of-distribution** (OOD) samples.

- **Infer latent variables** $$z$$. In the dog example we could understand the underlying common structure of dog images. These latent variables could be dog position, fur color, ears type...

**NB**: _Quantitative evaluation of generative models is non-trivial and is still being researched on._

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
TODO: Explanation $$p_\theta(x) = \prod_i p_\theta(x_i \mid x_{< i})$$.

Use of Variational INference approximation trick to avoid the integration.

Provide <span style="color:green">tractable likelihoods</span> but <span style="color:red">no direct mechanism for learning features</span>.

### Variational autoencoders (VAE)

TODO: Explanation
$$p_\theta(x) = \int p_\theta(x, z) dz = \int p_\theta(x \mid z) p(z) dz$$.
Where $$p_\theta(x \mid z)$$ is modelled by the decoder network and $$p(z)$$ the chosen prior for the latent variables $$z$$.

Can <span style="color:green">learn feature representations</span> $$(z)$$ but <span style="color:red">have intractable marginal likelihood</span> $$p_\theta(x \mid z)$$.

### Normalizing flow models

The main idea is to learn a deterministic bijective (invertible) **mapping** from **easy distributions** (easy to sample and easy to evaluate density, e.g. Gaussian) to the **given data distribution** (more complex).

First we need to understand the **change of variables formula**: Given $$Z$$ and $$X$$ random variables related by a bijective (invertable) mapping $$f : \mathbb{R}^n \rightarrow \mathbb{R}^n$$ such that $$X = f(Z)$$ and $$Z = f^{-1}(X)$$ then:

\begin{equation}
p_X(x) = p_Z \left( f^{-1} (x) \right) \left|\det \left( \frac{\partial f^{-1} (x)}{\partial x} \right)\right|
\end{equation}

Were $$\frac{\partial f^{-1} (x)}{\partial x}$$ is the $$n \times n$$ Jacobian matrix of $$f^{-1}$$.
Notice that its determinant models the **local** change of volume of $$f^{-1}$$ at the evaluated point.

**NB:** _"**Normalizing**" because the change of variables gives a normalized density after applying the transformations. "**Flow**" because the invertible transformations can be composed with each other to create more complex invertible transformations: $$f = f_0 \circ ... \circ f_k$$._

As you might have guessed, normalizing flow models parametrize this $$f$$ mapping function using an ANN $$(f_\theta)$$.
**This ANN**, however, needs to verify some specific architectural structures:

- <span style="color:red">Needs to be **deterministic**</span>
- <span style="color:red">I/O **dimensions** must be the **same** ($$f$$ has to be bijective)</span>
- <span style="color:red">Transformations must be **invertible**</span>
- <span style="color:red">Computation of the determinant of the Jacobian must be **efficient** and **differentiable**.</span>

Nevertheless they solve both previous approach problems:
- <span style="color:green">Present feature learning</span>.
- <span style="color:green">Present a tractable marginal likelihood</span>.

Most famous normalizing flow architectures ([NICE](https://arxiv.org/abs/1410.8516), [RealNVP](https://arxiv.org/abs/1605.08803), [Glow](https://arxiv.org/abs/1807.03039)) design layers whose Jacobian matrices are triangular or can be decomposed in triangular shape. These layers include variations of the **affine coupling layer**, **activation normalization layer** or **Invertible 1x1 conv**.

## Likelihood free learning

These models are not trained using maximum likelihood.

### Generative Adversarial Networks (GANs)