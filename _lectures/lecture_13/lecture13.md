---
layout: lecture
title: "Lecture 13: Variational Inference and Generative Models"
permalink: /lectures/lecture13
lecture-author: Sergey Levine
lecture-date: 2019
slides-link: http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-13.pdf
video-link: https://www.youtube.com/watch?v=1bpQ0QDPGuI&feature=youtu.be
---
<!--
Disclaimer and authorship:
This article is provided for free only for your personal informational and entertainment purposes. No commercial use of it is allowed.

Please note there might be mistakes. We would be grateful to receive (constructive) criticism if you spot any. You can reach us at: ai.campus.ai@gmail.com or directly open an issue on our github repo: https://github.com/CampusAI/CampusAI.github.io

If considering to use the text please cite the original author/s of the lecture/paper.
Furthermore, please acknowledge our work by adding a link to our website: https://campusai.github.io/ and citing our names: Oleguer Canal and Federico Taschin.
-->

In [Lecture 2: Imitation Learning](/lectures/lecture2) we noticed an issue with learning from
demonstrations when the data has a **multi-modal behavior**. In the example below, the human
demonstrator sometimes decides to avoid the obstacle by turning left, and sometimes by turning
right. However, averaging between the two will make you crash.

{% include figure.html url="/_lectures/lecture_13/tree.png" description="Example of bimodal behavior" %}

If our neural network outputs a Gaussian action, it will be forced to average between the human
demonstrations. We learned that there are three main ways of dealing with this issue:

- Output a mixture of Gaussians
- **Latent Variable Models**
- Autoregressive Discretization

In this lecture we talk about **Latent Variable Models**, and how to approximate any multimodal
distribution using **Variational Inference**.

It is often the case that data is distributed accordingly to some variables that cannot be directly observed. One schoolbook examples is the Gaussian Mixture:

{% include figure.html url="/_lectures/lecture_13/gauss_mix.png" description="Gaussian Mixture"%}

The likelyhood of a datapoint $$x$$ is given by the marginalization over the possible values of a
**latent variable** $$z \in \{1, 2, 3\}$$ that indicates the cluster.

$$
p(x) = \sum_{i=1}^3 p(x| z=i)p(z=i)
$$


## Latent Variable Models

The general formula of a Latent Variable Model with a latent variable $$z$$ is

\begin{equation}
\label{eq:lvm}
p(x) = \int p(x | z) p(z) dz
\end{equation}

and for a **conditioned** latent variable model we have

\begin{equation}
\label{eq:lvm_cond}
p(y | x) = \int p(y | x, z)p(z)
\end{equation}

Dealing with these integrals in practice is not easy at all, as for many complex distributions they
can be hard or impossible to compute. In this lecture we will learn how to approximate them.

If we want to represent a really complex distribution, we can represent $$p(x | z)$$ with a **Neural
Network**:

{% include figure.html url="/_lectures/lecture_13/nn_transform.png" description="Neural Network mapping $z$ to $p(x|z)$" %}

Note that $$p(x|z)$$ is a Gaussian, but the mean and variance of this Gaussian are given by the
non-linear function of the Neural Network, and therefore it can approximate any distribution.
