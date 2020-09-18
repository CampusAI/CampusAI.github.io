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

### Layers
Remember than in normalizing flows we look for [**bijective**](https://en.wikipedia.org/wiki/Bijection) (invertible), **deterministic**, **differentiable** operations with an easy to compute **Jacobian determinant**.

#### Activation normalization (actnorm)
Applies a scale $$s_c$$ and bias $$b_c$$ parameter per channel $$c$$ of the input. It is very similar to a [batch normalization](https://en.wikipedia.org/wiki/Batch_normalization) layer.
The parameters are learnt (when maximizing flow likelihood) but they are initialized so that they set 0 mean and 1 std dev to the first data batch (**data-dependent initialization**).

#### Affine coupling layers

Given a D-dimensional input $$x$$, this layer applies the following operations:

1. Split $$x$$ into 2 sub-vectors: $$x_{1:d}, x_{d+1:D}$$
2. The first part $$(x_{1:d})$$ is left untouched.
3. The second part undergoes an **affine transformation** (scale $$s$$ and translation $$t$$). This scale and translations are a function dependent on the first part of the input vector: $$s(x_{1:d}), t(x_{1:d})$$.

Overall:

\begin{equation}
y_{1:d} = x_{1:d}
\end{equation}

\begin{equation}
y_{d+1:D} = x_{d+1:D} \circ \exp (s(x_{1:d})) + t(x_{1:d})
\end{equation}

The inverse is trivial and its Jacobian a lower triangular matrix (log-determinant is then the addition of the log-elements of the diagonal).

**NB**: _$$s$$ and $$t$$ can be any function. In practice they are parametrized by an ANN._

**NB**: _The design of this layer is copied from [RealNVP](https://arxiv.org/abs/1605.08803), which adds the scale term on the coupling layer presented in [NICE](https://arxiv.org/abs/1410.8516) called **additive coupling**._

To ensure each dimension can affect each other dimension, we need to perform enough steps of some kind of **permutations** on the inputs. Previous work implemented a fixed simple permutation layer, this work generalizes this permutation by introducing the invertible 1x1 convolution.

#### Invertible 1x1 2D-convolution

This layer is a generalization of a permutation along the channels dimension.
Given an input of shape $$h\times w \times c$$.
It applies a 1x1 convolution with $$c$$ filters, meaning the output tensor shape is also going to be $$h\times w \times c$$ (we need to keep the dimensions in normalizing flows).

Thus, each layer has a set of weights $$W$$ with $$c \cdot c$$ values.
Its log-determinant can be computed as:

\begin{equation}
\log \left| det \left( \frac{\partial conv2D(x; W)}{\partial x} \right) \right| = h\cdot w \cdot \log| (W) |
\end{equation}

Its inverse operation can be computed by simply applying a convolution with $$W^{-1}$$ weights.

**NB**: _Usually $$c=3$$ so the $$det(W)$$ and $$W^{-1}$$ are cheap to compute._

### Layers summary
In summary this is each layer function, its inverse and its Jacobian log-determinant:

{% include figure.html url="/_papers/Glow/layers.png" description="Figure 2: Layer summary" zoom="1.0"%}

## Results

### Quantitative

The authors first show that:

- Using the Invertible 1x1 2D-convolution achieves lower NLL scores when compared to fixed shuffles 
and reverse layers.

- Affine coupling achieves lower NNL compared to additive coupling.

- Glow achieves around 8.5% lower NNL compared to [RealNVP](https://arxiv.org/abs/1605.08803) averaged around common image datasets.

### Qualitative

They test their algorithm on [CelebA-HQ](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset (30000 celeb images of $$256^2$$ px).

#### Synthesis and Interpolation

{% include figure.html url="/_papers/Glow/interpolation.png" description="Figure 3: Smooth interpolation in latent space between two real images examples." zoom="1.0"%}

#### Semantic Manipulation

Each image has a binary label indicating the presence or absence of some attribute.
For instance whether the celeb is smiling or not.
Let $$z_{pos}$$ be the average latent space position of samples presenting the attribute, and $$z_{neg}$$ the average of the samples which do not present it.
we can use the difference vector $$z_{pos} - z_{neg}$$ as a direction on which to manipulate the samples to modify that particular attribute:

{% include figure.html url="/_papers/Glow/attributes.png" description="Figure 4: Attribute manipulation examples." zoom="1.0"%}

#### Temperature

When sampling they add a temperature parameter $$T$$ and modify the distribution such that: $$p_{\theta, T} \propto \left( p_\theta (x)\right)^2$$.
They claim lower $$T$$ values provide higher-quality samples:

{% include figure.html url="/_papers/Glow/interpolation.png" description="Figure 5: Temperature parameter effect." zoom="1.0"%}

## Contribution

- Design of a new normalizing-flow layer: The **invertible 1x1 convolution**.

- Improved quantitative SOTA results (in terms of log-likelihood of the modelled distribution $$P(X)$$).

## Weaknesses

- As this paper is more an **evolution** rather than a **revolution** there are no major weaknesses. Future lines of work could focus on the design of new normalizing flow layers and experiment on those.

- They could further investigate the role of the temperature parameter $$T$$.