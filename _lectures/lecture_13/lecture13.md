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
p(y \vert  x) = \int p(y \vert  x, z)p(z) dz
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

the Maximum Likelihood fit to train the latent variable model $$p_{\theta}(x)$$ finds the parameters which better explain the data:
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
[Annex 13: Variational Inference](/lectures/variational_inference_annex),
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
the **Evidence Lower Bound**, shortened in ELBO. A useful mathematical tool to measure the
distance between two distributions is the
[KL Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence),
which is defined as
\begin{equation}
D_{KL}\Big(q_i(z) \vert\vert p(z \vert x_i)\Big) =
\int p(z \vert x_i) \ln \frac{p(z \vert x_i)}{q_i(z)}
\end{equation}
If we develop on this definition we obtain that
\begin{equation}
\label{eq:dkl}
\ln p(x_i) = D_{KL}(q_i(z) \vert\vert p(z \vert x_i)) + \mathcal{L}_i(p, q_i)
\end{equation}
*see the [Annex](/lectures/variational_inference_annex) for more details.*

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
new network. This network $$q_{\phi}(z \vert x)$$ will output the parameters of the distribution,
for example the mean and the variance of a Gaussian:
$$q_{\phi}(z \vert x) = \mathcal{N}(\mu_{\phi}(x), \sigma_{\phi}(x))$$. Since mean and variance
are given by the Neural Network, $$q_{\phi}$$ can approximate any distribution.

The ELBO $$\mathcal{L}_i(p, q)$$ is the same as before, with $$q_{\phi}$$ instead
of $$q_i$$:

\begin{equation}
\label{eq:amortized_elbo}
\mathcal{L_i}(p, q) = E_{z \sim q_{\phi}(z \vert x_i)}\Big[\ln p_{\theta}(x_i \vert z)
+\ln p(z) \Big] + \mathcal{H}\Big(q_{\phi}(z \vert x_i)\Big)
\end{equation}

We now have two networks: $$p_{\theta}$$ that learns $$p(x \vert z)$$, and $$q_{\phi}$$, that
approximates $$p(z \vert x)$$. We then modify the algorithm like this:

1. For each $$x_i$$ (or minibatch):
2. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sample $$z \sim q_{\phi}(z \vert x_i)$$
3. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Compute $$\nabla_{\theta}\mathcal{L}_i(p, q) \approx
   \nabla_{\theta}\ln p_{\theta}(x_i \vert z)$$ 
4. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$$\theta \leftarrow \theta + \alpha
   \nabla_{\theta}\mathcal{L}_i(p, q)$$
5. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$$\phi \leftarrow \phi + \alpha
   \nabla_{\phi}\mathcal{L}_i(p, q)$$

We now need to compute the gradient of Eq. \ref{eq:amortized_elbo} with respect to $$\phi$$:

\begin{equation}
\nabla_{\phi}\mathcal{L_i}(p, q) = \nabla_{\phi}E_{z \sim q_{\phi}(z \vert x_i)}\Big[
\overbrace{\ln p_{\theta}(x_i \vert z)+\ln p(z)}^{r(x_i, z)\text{, constant in } \phi} \Big] +
\nabla_{\phi}\mathcal{H}\Big(q_{\phi}(z \vert x_i)\Big)
\end{equation}

While the gradient of the entropy $$\mathcal{H}$$ can be computed straightforward by looking at
the formula in a textbook, the gradient of the expectation is somewhat trickier, since we need
to take the gradient of the parameters of the distribution under which the expectation is taken.
This is however exactly the same thing we do in **Policy Gradient**!
(see [Lecture 5: Policy Gradients](/lectures/lecture5)). Collecting the terms that do not depend
on $$\phi$$ under $$r(x_i, z) = \ln p_{\theta}(x_i \vert z) + \ln p(z)$$ we obtain the policy
gradient:

\begin{equation}
\nabla_{\phi}E_{z \sim q_{\phi}(z\vert x_i)} \left[r(x_i, z)\right]
= \frac{1}{M} \sum_{j=1}^M \nabla_{\phi}\ln q_{\phi}(z_j \vert x_i)r(x_i, z_j)
\end{equation}

where we estimate the gradient by averaging over $$M$$ samples
$$z_j \sim q_{\phi}(z \vert x_i)$$. We therefore obtain the gradient of $$\mathcal{L}_i(p, q)$$
of Eq. \ref{eq:amortized_elbo}:

\begin{equation}
\label{eq:elbo_pgradient}
\boxed{
\nabla_{\phi}\mathcal{L_i}(p, q)
= \frac{1}{M} \sum_{j=1}^M \nabla_{\phi}\ln q_{\phi}(z_j \vert x_i)r(x_i, z_j)
+\nabla_{\phi}\mathcal{H}\Big[q_{\phi}(z \vert x_i)\Big]
}
\end{equation}

#### Reducing variance: The reparametrization trick
The formula for the ELBO gradient we found suffers from the same problem of simple Policy
Gradient: the high variance. Assuming the network $$q_{\phi}$$ outputs a Gaussian distribution
$$z \sim \mathcal{N}(\mu_{\phi}(x), \sigma_{\phi}(x))$$, then $$z$$ can be written as

\begin{equation}
\label{eq:z_rep_trick}
z = \mu_{\phi}(x) + \epsilon \sigma_{\phi}(x)
\end{equation}
where $$\epsilon \sim \mathcal{N}(0, 1)$$. Now, the first term of Eq. \ref{eq:amortized_elbo}
can be written as an expectation over the standard gaussian, and $$z$$ substituted with Eq.
\ref{eq:z_rep_trick}.
\begin{equation}
E_{z \sim q_{\phi}(z \vert x_i)}\Big[r(x_i, z)\Big] =
E_{\epsilon \sim \mathcal{N(0, 1)}}\Big[r(x_i, \mu_{\phi}(x_i)+\epsilon\sigma_{\phi}(x_i))\Big]
\end{equation}

Now, the parameter $$\phi$$ does not appear anymore in the distribution, but rather in the
optimization objective. We can therefore take the gradient by sampling $$M$$ values of
$$\epsilon$$:

\begin{equation}
\nabla_{\phi} E_{\epsilon \sim \mathcal{N(0, 1)}}
\Big[r(x_i, \mu_{\phi}(x_i)+\epsilon\sigma_{\phi}(x_i))\Big] = 
\frac{1}{M} \sum_{j=1}^M \nabla_{\phi}r(x_i, \mu_{\phi}(x_i)+\epsilon_j\sigma_{\phi}(x_i))
\end{equation}

Note that now gradient flows directly into $$r$$. This improves the gradient estimation, but
requires the $$q_{\phi}$$ network to output a distribution that allows us to use this trick.
In practice, this gradient has low variance, and **a single sample of $$\epsilon$$ is sufficient
to estimate it**. Using the reparametrization trick, the full gradient becomes:

\begin{equation}
\label{eq:elbo_trick_gradient}
\boxed{
\nabla_{\phi}\mathcal{L_i}(p, q)
= \frac{1}{M} \sum_{j=1}^M \nabla_{\phi}r(x_i, \mu_{\phi}(x_i)+\epsilon_j\sigma_{\phi}(x_i))
+\nabla_{\phi}\mathcal{H}\Big[q_{\phi}(z \vert x_i)\Big]
}
\end{equation}

The difference between this gradient and that of Eq. \ref{eq:elbo_pgradient} is that while here
we are able to use the gradient of $$r$$ directly, in Eq. \ref{eq:elbo_pgradient} we rely
on the gradient of $$q_{\phi}$$ in order to increase the likelihood of $$x_i$$ that make $$r$$
large. This is the same we did in [Lecture 5: Policy Gradients](/lectures/lecture5), where we
discussed why doing this leads to an high variance estimator. The figure below shows the
process that from $$x_i$$ gives us $$q_{\phi}(z \vert x_i)$$ and $$p_{\theta}(x_i \vert z)$$:
{% include figure.html url="/_lectures/lecture_13/icon.png" description="" %}


#### A more practical form of $$\mathcal{L_i}(p, q)$$

If we look at Eq. \ref{eq:amortized_elbo}, we observe that it can be written in terms of the
KL Divergence between $$q_{\phi}$$ and $$p(z)$$:

\begin{equation}
\mathcal{L_i}(p, q) = E_{z \sim q_{\phi}(z \vert x_i)}\Big[\ln p_{\theta}(x_i \vert z)\Big]+
\overbrace{
E_{z \sim q_{\phi}(z \vert x_i)}\Big[\ln p(z) \Big]
+\mathcal{H}\Big(q_{\phi}(z \vert x_i)\Big)
}^{-D_{KL}\Big(q_{\phi}(z \vert x_i) \vert\vert p(z) \Big)}
\end{equation}
In practical implementations is often better to group the last two terms under the KL DIvergence
since we can compute it analytically -e.g.
[D. Kingma, M. Welling, Auto Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf)-,
and we can use the reparametrization trick only on the first term.

The **Policy Gradient** for $$\mathcal{L_i}(p, q)$$ with respect to $$\phi$$ then becomes:
\begin{equation}
\label{eq:elbo_pgradient_dkl}
\boxed{
\nabla_{\phi} \mathcal{L_i}(p, q) =
\frac{1}{M} \sum_{j=1}^M \nabla_{\phi}q_{\phi}(z_j \vert x_i)\ln p_{\theta}(x_i \vert z_j)
-\nabla_{\phi}D_{KL}\Big(q_{\phi}(z \vert x_i) \vert \vert p(z)\Big)
}
\end{equation}

The **Reparametrized Gradient** then becomes (single sample estimate):
\begin{equation}
\label{eq:elbo_trick_gradient_dkl}
\boxed{
\nabla_{\phi} \mathcal{L_i}(p, q) = 
\nabla_{\phi} \ln p_{\theta}(x_i \vert \mu_{\phi}(x_i) + \epsilon \sigma_{\phi}(x_i)) -
\nabla_{\theta} D_{KL}\Big(q_{\phi}(z \vert x_i) \vert\vert p(z)\Big)
}
\end{equation}

### Policy Gradient or Reparametrization Trick?

**Policy Gradient** (Eq. \ref{eq:elbo_pgradient_dkl}):
+ <span style="color:green">Can handle both discrete and continuous latent variables</span>.
+ <span style="color:red">High variance, requires multiple samples and smaller
learning rates</span>.

**Reparametrized Gradient** (Eq. \ref{eq:elbo_trick_gradient_dkl}):
+ <span style="color:green">Low variance (one sample is often enough)</span>.
+ <span style="color:green">Simple to implement</span>.
+ <span style="color:red">Can handle only continuous latent variables</span>.
