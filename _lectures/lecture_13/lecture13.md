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
p(x) = \sum_{i=1}^3 p(x\vert  z=i)p(z=i)
$$


## Latent Variable Models

The general formula of a Latent Variable Model with a latent variable $$z$$ is obtained by
marginalization:

\begin{equation}
\label{eq:lvm}
p(x) = \int p(x \vert  z) p(z) dz
\end{equation}

and for a **conditioned** latent variable model we have:

\begin{equation}
\label{eq:lvm_cond}
p(y \vert  x) = \int p(y \vert  x, z)p(z)
\end{equation}

Dealing with these integrals in practice is not easy at all, as for many complex distributions they
can be hard or impossible to compute. In this lecture we will learn how to approximate them.

If we want to represent a really complex distribution, we can represent $$p(x \vert  z)$$ with a **Neural Network** that, given $$z$$, will output the mean and variance of a Gaussian
distribution for $$x$$:

{% include figure.html url="/_lectures/lecture_13/nn_transform.png" description="Neural Network mapping $z$ to $p(x\vert z)$" %}

Note that $$p(x\vert z)$$ is a Gaussian, but the mean and variance of this Gaussian are given
by the non-linear function of the Neural Network, and therefore it can approximate any
distribution. Given a dataset

$$
\mathcal{D} = \left\{ x_1, x_2, \: ... \:x_N\right\}
$$

the Maximum Likelyhood fit to train the latent variable model $$p_{\theta}(x)$$ is
\begin{equation}
\label{eq:ml_lvm}
\theta \leftarrow \arg\max_{\theta} \frac{1}{N}\sum_{i=1}^N \log p_{\theta}(x_i) =
\arg\max_{\theta} \frac{1}{N}\sum_{i=1}^N \log \left(\int p_{\theta}(x_i \vert z)p(z)dz\right)
\end{equation}
The integral makes the computation intractable, therefore we need to resort to other ways of
computing the log likelihood.


## Estimating the log likelihood

Eq. \ref{eq:ml_lvm} requires us to compute $$p_{\theta}(x)$$, which involves integrating the
latent variables and is therefore intractable. One important technique for finding Maximum
Likelihood solutions to latent variable models is **Expectation Maximization**
(see chapter 9 of [C. Bishop, Pattern Recognition and Machine Learning](https://www.microsoft.com
/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)).

Expectation Maximization consists of iteratively alternating between the following steps:

1. **E step**: Compute the posterior $$p_{\theta^{old}}(z \vert x_i)$$
2. **M step**: Maximize the expected value of the log joint likelihood 
   $$E_{z \sim p_{\theta^{old}}(z\vert x_i)} \Big[ \ln p_{\theta}(x_i, z) \Big]$$
   over the parameters $$\theta$$


Eq. \ref{eq:ml_lvm} then becomes
\begin{equation}
\label{eq:ml_lvm_em}
\theta \leftarrow \arg\max_{\theta} \frac{1}{N}\sum_{i=1}^N
E_{z \sim p(z\vert x_i)} \Big[ \ln p_{\theta}(x_i, z) \Big]
\end{equation}

but we now have the issue of computing $$p(z \vert x_i)$$, which is likely to be intractable.
Instead, we **approximate** $$p(z \vert x_i)$$ with $$q_i(z)$$ using **Variational Inference**.


### Variational Inference

*For a deeper explanation of this very useful tool, see Chapter 10.1 of
[C. Bishop, Pattern Recognition and Machine Learning](https://www.microsoft.com
/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)*

We are interested in the maximization of Eq. \ref{eq:ml_lvm_em}, but we need to
perform it by approximating $$p(z\vert x_i)$$ that is intractable with a distribution
$$q_i(z)$$ that we choose to be tractable. $$q_i(z)$$ can be any analytically parametrized
distribution. As we show in
[Annex 13: Variational Inference](/lectures/annex/variational_inference),
the log of $$p(x)$$ is bounded by:

\begin{align}
\label{eq:elbo}
\begin{split}
\ln p(x_i) \ge &  E_{z \sim q_i(z)} \Big[\overbrace{\ln p_{\theta}(x_i\vert z)+\ln p(z)}
^{\ln p(x_i, z)}\Big]+
\mathcal{H}(q_i) \\\\\\
= & \mathcal{L}_i(p, q_i)
\end{split}
\end{align}


where $$\mathcal{H}(q_i)$$ is the **entropy** of $$q_i$$ and $$\mathcal{L}_i(p, q_i)$$ is called
the **Evidence Lower Bound**, shortened in ELBO. Moreover, if we develop on the definition of
KL Divergence, we obtain that
\begin{equation}
\label{eq:dkl}
\ln p(x_i) = D_{KL}(q_i(z) \vert\vert p(z \vert x_i)) + \mathcal{L}_i(p, q_i)
\end{equation}
*see the [Annex](/lectures/annex/variational_inference) for more details.*

The two results together give us a way to approximate $$p(x_i)$$: 

1. **Maximize $$p(x_i)$$ w.r.t. $$\theta$$**: As Eq. \ref{eq:elbo} shows, we can maximize
   $$p(x_i)$$ with respect to $$\theta$$ by maximizing the ELBO $$\mathcal{L}_i(p, q_i)$$.
2. **Maximize $$p(x_i)$$ w.r.t. $$q_i$$**: Since the KL Divergence is always greater than zero,
   we can exploit Eq. \ref{eq:dkl}, that shows us that minimizing the $$D_{KL}$$ term brings
   the equation closer to the equality, i.e. $$\ln p(x_i) \approx \mathcal{L}_i(p, q_i)$$. 
   Minimizing the $$D_{KL}$$ term corresponds to maximizing the Evidence Lower Bound
   $$\mathcal{L}_i(p, q_i)$$.

We obtained an important result: in order to maximize $$p(x_i)$$, we need to maximize the
**Evidence Lower Bound** of Eq. \ref{eq:elbo} both with respece to $$\theta$$ and $$q_i$$.
Our goal is therefore to find $$\theta^*$$ such that:
\begin{equation}
\theta^* = \arg\max_{\theta} \frac{1}{N}\sum_{i=1}^N \mathcal{L}_i(p, q_i)
\end{equation}
that we optimize with the following algorithm:

1. For each $$x_i$$ (or minibatch):
2. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sample $$z \sim q_i(z)$$
3. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Compute $$\nabla_{\theta}\mathcal{L}_i(p, q_i) \approx
   \nabla_{\theta}\ln p_{\theta}(x_i \vert z)$$ 
4. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$$\theta \leftarrow \theta + \alpha
   \nabla_{\theta}\mathcal{L}_i(p, q_i)$$
5. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update $$q_i$$ to maximize $$\mathcal{L}_i(p, q_i)$$

There are many things to notice about what we just did. First, when we update $$\theta$$ in line
4 we are performing the EM maximization of Eq. \ref{eq:ml_lvm_em} with $$p(z \vert x_i)$$
approximated by $$q_i(z)$$. In fact, the gradient w.r.t. $$\theta$$ acts only on the first
expectation of Eq. \ref{eq:elbo} since the entropy does not depend on $$\theta$$. The sampling
step of line 2 is required because we are optimizing an expectation over $$q_i(z)$$ which we
are not able to compute, and we therefore estimate its gradient by sampling. Finally, we
update $$q_i$$ by maximizing the ELBO $$\mathcal{L}_i(p, q_i)$$, which in Eq. \ref{eq:dkl} we
showed being equivalent to minimizing the KL Divergence between $$q_i(z)$$ and
$$p(z \vert x_i)$$ and thus pushing $$q_i(z)$$ closer to $$p(z\vert x_i)$$.

#### What is the issue?
We said that $$q_i(z)$$ approximates $$p(z\vert x_i)$$. This means that if our dataset has $$N$$
datapoints, we would need to maximize $$N$$ approximate distributions $$q_i$$. For any large
dataset, such as those generated in Reinforcement Learning, we would end up having more
parameters for the approximate distributions than in our Neural Network!


### Amortized Variational Inference
As we said, having a distribution $$q_i$$ for each datapoint can lead us with an extreme number
of parameters. We therefore employ another **Neural Network** to approximate $$p(z \vert x_i)$$
with a contained number of parameters. We denote with $$\phi$$ the set of parameters of this
new network. The ELBO $$\mathcal{L}_i(p, q)$$ is now:

$$
\mathcal{L}_{i}(p, q) = E_{z \sim q_{\phi}(z \vert x_i)} \Big[\ln p_{\theta}(x_i \vert z)
+ \ln p(z) \Big] + \mathcal{H}\Big(q_{\phi}(z \vert x_i)\Big)
$$

and the algorithm now looks like this:

1. For each $$x_i$$ (or minibatch):
2. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sample $$z \sim q_{\phi}(z \vert x_i)$$
3. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Compute $$\nabla_{\theta}\mathcal{L}_i(p, q) \approx
   \nabla_{\theta}\ln p_{\theta}(x_i \vert z)$$ 
4. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$$\theta \leftarrow \theta + \alpha
   \nabla_{\theta}\mathcal{L}_i(p, q)$$
5. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$$\phi \leftarrow \phi + \alpha
   \nabla_{\phi}\mathcal{L}_i(p, q)$$

## Conclusions
In the paragraphs above we learned how to approximate any distribution by introducing some
latent variables and a neural network. We then derived an algorithm to approximate the
distribution of a dataset $$D$$ 
