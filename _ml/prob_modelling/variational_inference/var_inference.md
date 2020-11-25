---
layout: lecture
title: "LVM: EM & VI"
permalink: /ml/variational_inference
lecture-author: Sergey Levine
lecture-date: 2019
post-author: Federico Taschin, Oleguer Canal
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
{% include start-row.html %}

<!-- In [Lecture 2: Imitation Learning](/lectures/lecture2) we noticed an issue with learning from
demonstrations when the data has a **multi-modal behavior**. In the example below, the human
demonstrator sometimes decides to avoid the obstacle by turning left, and sometimes by turning
right. However, averaging between the two will make you crash.

{% include figure.html url="/_ml/prob_modelling/variational_inference/tree.png" description="Example of bimodal behavior" %}

If our neural network outputs a Gaussian action, it will be forced to average between the human
demonstrations. We learned that there are three main ways of dealing with this issue:

- Output a mixture of Gaussians
- **Latent Variable Models**
- Autoregressive Discretization -->

In this post we talk about **Latent Variable Models**, and how to approximate any multimodal distribution using **Variational Inference**.

It is often the case that data is distributed accordingly to some variables that cannot be directly observed. One schoolbook examples is the Gaussian Mixture:

{% include figure.html url="/_ml/prob_modelling/variational_inference/gauss_mix.png" description="Gaussian Mixture"%}

The likelihood of a datapoint $$x$$ is given by the marginalization over the possible values of a
**latent variable** $$z \in \{1, 2, 3\}$$ that indicates the cluster.

$$
p(x) = \sum_{i=1}^3 p(x\vert  z=i)p(z=i)
$$

## Latent Variable Models (LVM)

Usually though, modelling distributions with low-range discrete latent variables is not good enough.
The general formula of a Latent Variable Model with a latent variable $$z$$ is obtained by marginalization:

\begin{equation}
\label{eq:lvm}
p(x) = \int p(x \vert  z) p(z) dz
\end{equation}

{% include end-row.html %}
{% include start-row.html %}

and for a **conditioned** latent variable model we have:

\begin{equation}
\label{eq:lvm_cond}
p(y \vert  x) = \int p(y \vert  x, z)p(z) dz
\end{equation}

{% include annotation.html %}
This is usually the case when modelling distributions with ANNs, given $$x$$, one wants to know $$p(y \mid x)$$.
Which can be explained by marginalizing over some simpler distribution $$z$$. 
{% include end-row.html %}
{% include start-row.html %}

Dealing with these integrals in practice is not easy at all, as for many complex distributions they
can be hard or impossible to compute. In this lecture we will learn how to approximate them.

{% include end-row.html %}
{% include start-row.html %}

If we want to represent a really complex distribution, we can represent $$p(x \vert  z)$$ with a **Neural Network** that, given $$z$$, will output the mean and variance of a Gaussian distribution for $$x$$:

{% include figure.html url="/_ml/prob_modelling/variational_inference/nn_transform.png" description="Neural Network mapping $z$ to $p(x\vert z)$" %}

{% include annotation.html %}
You can imagine this as a mixture of infinite Gaussians:
For every possible value of $$z$$, a Gaussian $$\mathcal{N}(\mu_{nn}(z), \sigma_{nn}(z))$$ is summed to the approximation of $$p(x)$$.

Note that $$p(x\vert z)$$ is a Gaussian, but the mean and variance of this Gaussian are given by the non-linear function of the Neural Network, and therefore it can approximate any distribution.
{% include end-row.html %}
{% include start-row.html %}


### How do you train latent variable models?

{% include end-row.html %}
{% include start-row.html %}
So, given a dataset $$\mathcal{D} = \left\{ x_1, x_2, \: ... \:x_N\right\}$$ we want to learn the underlying distribution $$p(x)$$.
Since we are using latent variable models, we suspect there exists some simpler distribution $$z$$ (usually Gaussian) which can be used to explain $$p(x)$$.
 
{% include annotation.html %}
For further motivations on why one would want to learn $$p(x)$$ check out our post: [Why generative models?](/ml/generative_models)
{% include end-row.html %}
{% include start-row.html %}

One can parametrize $$p(x)$$ with some function $$p_\theta (x)$$ adn find the parameters $$\theta$$ which minimizes the distance between the two.
The Maximum Likelihood fit of $$p_{\theta}(x)$$ finds the parameters $$\theta$$ which better explain the data.
I.e. the $$\theta$$ which give a higher probability to the given dataset:

{% include end-row.html %}
{% include start-row.html %}

\begin{equation}
\label{eq:ml_lvm}
\theta \leftarrow \arg\max_{\theta} \frac{1}{N}\sum_{i=1}^N \log p_{\theta}(x_i) =
\arg\max_{\theta} \frac{1}{N}\sum_{i=1}^N \log \left(\int p_{\theta}(x_i \vert z)p(z)dz\right)
\end{equation}

The integral makes the computation intractable, therefore we need to resort to other ways of computing the log likelihood.

{% include annotation.html %}
Why do we maximize the probability of each data-point?
Since we assume our dataset is composed by samples of $$p(x)$$, we want the samples to be very likely, thus we maximize their probability.
{% include end-row.html %}
{% include start-row.html %}

## Estimating the log likelihood

### Expectation Maximization (EM)
{% include end-row.html %}
{% include start-row.html %}

Eq. $$\ref{eq:ml_lvm}$$ requires us to compute $$p_{\theta}(x)$$, which involves integrating the latent variables (usually multi-dimensional) and is therefore intractable.
Lets develop a bit the expression and see if we can come up with something easier to deal with:

$$
\log p (x \mid \theta) =
\int_z \log p\left(x \mid z, \theta \right) p(z) = 
\int_z \log  \frac {p \left(x, z \mid \theta \right)}{p \left(z \mid x, \theta \right)} p(z) =
\underbrace{\int_z \log \frac {p \left(x, z \mid \theta \right)}{p(z)} p(z)}_{\mathcal{L}(p(z), \theta)} 
\underbrace{- \int_z \log \frac {p \left(z \mid x, \theta \right)}{p(z)} p(z)}_{D_{KL} \left(p(z) \Vert p(x \mid \theta) \right)}
$$

<!-- Since $$p(z)$$ is chosen and $$p(x)$$ is constant: $$KL \left(p(z) \Vert p(x \mid \theta) \right)$$ is constant. -->
Things to note:

- By definition: $$D_{KL} \left(p(z) \Vert p(x \mid \theta) \right) \ge 0$$

- Which means: $$\log p(X \mid \theta) \geq \mathcal{L}(p(z), \theta) := E_{z \sim p(z \vert x_i)} \left[ \log p_{\theta}(x_i, z) \right]$$

- Therefore we say: $$\mathcal{L}(p(z), \theta)$$ is a **lower bound** of $$\log p(X \mid \theta)$$

- Thus, by **maximizing** $$\mathcal{L}(p(z), \theta)$$ you will also **push up** $$\log p (x \mid \theta)$$

{% include annotation.html %}
{% include figure.html url="/_ml/prob_modelling/variational_inference/EM_interpretation.png" description="C. Bishop shows this decomposition with this figure. He represents $p(z)$ as $q$." %}
See chapter 9 of [C. Bishop, Pattern Recognition and Machine Learning](https://www.microsoft.com
/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
{% include end-row.html %}
{% include start-row.html %}
<!-- 
{% include end-row.html %}
{% include start-row.html %} -->

**Expectation Maximization** assumes $$p_{\theta^{old}}(z \vert x_i)$$ is tractable and exploits the lower bound inequality by iteratively alternating between the following steps:

<blockquote markdown="1">
Repeat until convergence:
1. **E step**: Compute the posterior $$p_{\theta^{old}}(z \vert x_i)$$
2. **M step**: Maximize the expected value of the log joint likelihood 
   $$E_{z \sim p_{\theta^{old}}(z\vert x_i)} \Big[ \log p_{\theta}(x_i, z) \Big]$$
   over the parameters $$\theta$$
</blockquote>

{% include annotation.html %}
Intuitively, what we are doing is:
1. Assume the probability of each $$z$$ given $$x_i$$ (the probability of each cluster if $$z$$ discrete).
2. Pretend $$z$$ is the right one and maximize the expected log-likelihood based on your guessed $$z$$.
{% include end-row.html %}
{% include start-row.html %}

Eq. \ref{eq:ml_lvm} then becomes the maximization of $$\mathcal{L}(p(z), \theta)$$:
\begin{equation}
\label{eq:ml_lvm_em}
\theta \leftarrow \arg\max_{\theta} \frac{1}{N}\sum_{i=1}^N
E_{z \sim p(z \vert x_i)} \left[ \log p_{\theta}(x_i, z) \right]
\end{equation}

While for discrete $$z$$ values (clusters of data) computing $$p(z \vert x_i)$$ might be tractable, this is not usually th case when mapping from continuous $$z$$ to continuous $$p(x)$$.
Instead, we **approximate** $$p(z \vert x_i)$$ with a simpler parametrized distribution $$q_i(z)$$ using **Variational Inference**.


### Variational Inference (VI)

{% include end-row.html %}
{% include start-row.html %}

As we said, we are interested in the maximization of Eq. \ref{eq:ml_lvm_em}, but $$p(z\vert x_i)$$ is intractable.
**Variational inference** approximates it using a tractable parametrization $$q_i(z) \simeq p(z\vert x_i)$$ dependent on $$\phi_i$$.

We thus have to optimize two sets of parameters:
- $$\theta$$ of $$p_{\theta}(x_i\vert z)$$
- $$\{ \phi_i \}_i$$ of $$q_i(z) \simeq p_{\theta}(z \vert x_i)$$

#### Optimizing $$\theta$$
As we show in [Annex 13: Variational Inference](/lectures/variational_inference_annex), the log of $$p(x)$$ is bounded by:

{% include annotation.html %}
For a deeper explanation see Chapter 10.1 of
[C. Bishop, Pattern Recognition and Machine Learning](https://www.microsoft.com
/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
{% include end-row.html %}
{% include start-row.html %}

{% include end-row.html %}
{% include start-row.html %}

\begin{align}
\begin{split}
\log p(x_i) \ge &  E_{z \sim q_i(z)} \Big[\overbrace{\log p_{\theta}(x_i\vert z)+\log p(z)}
^{\log p(x_i, z)}\Big]+
\mathcal{H}(q_i) \\\\\\
=: & \mathcal{L}_i(p, q_i)
\end{split}
\label{eq:elbo}
\end{align}

Where:

- $$\mathcal{H}(q_i)$$ is the **entropy** of $$q_i$$.

Things to note:
- Again, we have that: $$\log p(x_i) \ge \mathcal{L}_i(p, q_i)$$

- Thus, $$\mathcal{L}_i(p, q_i)$$ is called the **Evidence Lower Bound** (shortened as **ELBO**).

- Again, if you **maximize** this lower bound you will also **push up** the entire log-likelihood.

{% include annotation.html %}
See our [Information Theory Post](/ml/prob_modelling) for a better interpretation of Entropy $$\mathcal{H}$$ and KL Divergence.
(which we do not have access to).
{% include end-row.html %}
{% include start-row.html %}

<!-- The two results together give us a way to approximate $$p(x_i)$$:  -->

<!-- <blockquote markdown="1"> -->
<!-- 1. **Maximize $$p(x_i)$$ w.r.t. $$\theta$$**: As Eq. \ref{eq:elbo} shows, we can push up $$p(x_i)$$ with respect to $$\theta$$ by maximizing the ELBO $$\mathcal{L}_i(p, q_i)$$.

1. **Maximize $$p(x_i)$$ w.r.t. $$q_i$$**: Since the KL Divergence is always greater than zero, we can exploit Eq. \ref{eq:dkl}, that shows us that minimizing the $$D_{KL}$$ term brings the equation closer to the equality, i.e. $$\log p(x_i) \approx \mathcal{L}_i(p, q_i)$$. Minimizing the $$D_{KL}$$ term corresponds to maximizing the Evidence Lower Bound $$\mathcal{L}_i(p, q_i)$$. -->
<!-- </blockquote> -->

<!-- We obtained an important result: -->
<!-- Notice that both results are the same, maximizing **ELBO** is equivalent to minimizing $$D_{KL}$$:
The more similar your approximation $$q_i$$ is to   -->

<!-- In order to maximize $$p(x_i)$$, we need to maximize the **Evidence Lower Bound** of Eq. \ref{eq:elbo} both with respece to $$\theta$$ and $$q_i$$. -->
Our goal is therefore to find $$\theta^*$$ such that $$\mathcal{L}_i(p, q_i)$$ is maximized:

\begin{equation}
\theta^* = \arg\max_{\theta} \frac{1}{N}\sum_{i=1}^N
E_{z \sim q_i(z)} \Big[\log p_{\theta}(x_i\vert z)+\log p(z) \Big]+
\mathcal{H}(q_i)
\end{equation}

#### Optimizing $$\{\phi_i\}_i$$

Notice that you can also express $$p(x_i)$$ as:

\begin{equation}
\label{eq:dkl}
\log p(x_i) = D_{KL}(q_i(z) \vert\vert p(z \vert x_i)) + \mathcal{L}_i(p, q_i)
\end{equation}

Where, we can see how minimizing $$D_{KL}$$ is analogous to maximizing the **ELBO**:
We are looking to make $$q_i$$ as close as possible to the real $$p(z \mid x_i)$$, which will make $$D_{KL}$$ smaller and the **ELBO** bigger.

Therefore, apart from maximizing the **ELBO** wrt $$\theta$$ we should also maximize it wrt $$\phi_i$$.


#### VI algorithm

VI combines both optimizations in the following algorithm:
{% include end-row.html %}
{% include start-row.html %}

<blockquote markdown="1">
For each $$x_i$$ (or minibatch):
1. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sample $$z \sim q_i(z)$$
2. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Compute $$\nabla_{\theta}\mathcal{L}_i(p, q_i) \approx
   \nabla_{\theta}\log p_{\theta}(x_i \vert z)$$.
3. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$$\theta \leftarrow \theta + \alpha
   \nabla_{\theta}\mathcal{L}_i(p, q_i)$$
4. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update $$q_i$$ to maximize $$\mathcal{L}_i(p, q_i)$$ (Can be done using gradient descent on each $$\phi_i$$)
</blockquote>

{% include annotation.html %}
Why $$\nabla_{\theta}\mathcal{L}_i(p, q_i) \approx \nabla_{\theta}\log p_{\theta}(x_i \vert z)$$?
1. Since we cannot compute the expectation over $$q_i(z)$$, we estimate the expectation by sampling (Done in step 1.) 
2. The gradient w.r.t. $$\theta$$ acts only on the first expectation of Eq. \ref{eq:elbo} since the entropy does not depend on $$\theta$$.
{% include end-row.html %}
{% include start-row.html %}

Things to note:
- When we update $$\theta$$ in step 3. we are performing the EM maximization of Eq. \ref{eq:ml_lvm_em} with $$p(z \vert x_i)$$ approximated by $$q_i(z)$$.

- We update $$q_i$$ by maximizing the ELBO $$\mathcal{L}_i(p, q_i)$$, which in Eq. \ref{eq:dkl} we showed being equivalent to minimizing the KL Divergence between $$q_i(z)$$ and $$p(z \vert x_i)$$ and thus pushing $$q_i(z)$$ closer to $$p(z\vert x_i)$$.

<!-- In short, VI simplifies p(z | x_i) and solves the problem exactly for that simplification -->

#### What is the issue?
We said that $$q_i(z)$$ approximates $$p(z\vert x_i)$$. This means that if our dataset has $$N$$
datapoints, we would need to maximize $$N$$ approximate distributions $$q_i$$. For any large
dataset, such as those generated in Reinforcement Learning, we would end up having more
parameters for the approximate distributions than in our Neural Network!


### Amortized Variational Inference
Having a distribution $$q_i$$ for each datapoint can lead us with an extreme number
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
\mathcal{L_i}(p, q) = E_{z \sim q_{\phi}(z \vert x_i)}\Big[\log p_{\theta}(x_i \vert z)
+\log p(z) \Big] + \mathcal{H}\Big(q_{\phi}(z \vert x_i)\Big)
\end{equation}

We now have two networks: $$p_{\theta}$$ that learns $$p(x \vert z)$$, and $$q_{\phi}$$, that
approximates $$p(z \vert x)$$. We then modify the algorithm like this:

<blockquote markdown="1">
For each $$x_i$$ (or minibatch):
1. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sample $$z \sim q_{\phi}(z \vert x_i)$$
2. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Compute $$\nabla_{\theta}\mathcal{L}_i(p, q) \approx
   \nabla_{\theta}\log p_{\theta}(x_i \vert z)$$ 
3. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$$\theta \leftarrow \theta + \alpha
   \nabla_{\theta}\mathcal{L}_i(p, q)$$
4. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$$\phi \leftarrow \phi + \alpha
   \nabla_{\phi}\mathcal{L}_i(p, q)$$
</blockquote>

We now need to compute the gradient of Eq. \ref{eq:amortized_elbo} with respect to $$\phi$$:

\begin{equation}
\nabla_{\phi}\mathcal{L_i}(p, q) = \nabla_{\phi}E_{z \sim q_{\phi}(z \vert x_i)}\Big[
\overbrace{\log p_{\theta}(x_i \vert z)+\log p(z)}^{r(x_i, z)\text{, constant in } \phi} \Big] +
\nabla_{\phi}\mathcal{H}\Big(q_{\phi}(z \vert x_i)\Big)
\end{equation}

While the gradient of the entropy $$\mathcal{H}$$ can be computed straightforward by looking at
the formula in a textbook, the gradient of the expectation is somewhat trickier:
we need to take the gradient of the parameters of the distribution under which the expectation is taken.
This is however exactly the same thing we do in **Policy Gradient RL**! (see the **log gradient trick** in our [Policy Gradients Post](/lectures/lecture5)).
Collecting the terms that do not depend on $$\phi$$ under $$r(x_i, z) := \log p_{\theta}(x_i \vert z) + \log p(z)$$ we obtain:

\begin{equation}
\nabla_{\phi}E_{z \sim q_{\phi}(z\vert x_i)} \left[r(x_i, z)\right]
= \frac{1}{M} \sum_{j=1}^M \nabla_{\phi}\log q_{\phi}(z_j \vert x_i)r(x_i, z_j)
\end{equation}

where we estimate the gradient by averaging over $$M$$ samples
$$z_j \sim q_{\phi}(z \vert x_i)$$. We therefore obtain the gradient of $$\mathcal{L}_i(p, q)$$
of Eq. \ref{eq:amortized_elbo}:

\begin{equation}
\label{eq:elbo_pgradient}
\boxed{
\nabla_{\phi}\mathcal{L_i}(p, q)
= \frac{1}{M} \sum_{j=1}^M \nabla_{\phi}\log q_{\phi}(z_j \vert x_i)r(x_i, z_j)
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
optimization objective.
We can therefore take the gradient approximating the expectation by sampling $$M$$ values of $$\epsilon$$:

\begin{equation}
\nabla_{\phi} E_{\epsilon \sim \mathcal{N(0, 1)}}
\Big[r(x_i, \mu_{\phi}(x_i)+\epsilon\sigma_{\phi}(x_i))\Big] = 
\frac{1}{M} \sum_{j=1}^M \nabla_{\phi}r(x_i, \mu_{\phi}(x_i)+\epsilon_j\sigma_{\phi}(x_i))
\end{equation}

Note that now gradient flows directly into $$r$$.
This improves the gradient estimation, but requires the $$q_{\phi}$$ network to output a distribution that allows us to use this trick (e.g. Gaussian).
In practice, this gradient has low variance, and **a single sample of $$\epsilon$$ is sufficient to estimate it**.
Using the reparametrization trick, the full gradient becomes:

\begin{equation}
\label{eq:elbo_trick_gradient}
\boxed{
\nabla_{\phi}\mathcal{L_i}(p, q)
= \frac{1}{M} \sum_{j=1}^M \nabla_{\phi}r(x_i, \mu_{\phi}(x_i)+\epsilon_j\sigma_{\phi}(x_i))
+\nabla_{\phi}\mathcal{H}\Big[q_{\phi}(z \vert x_i)\Big]
}
\end{equation}

The difference between this gradient and that of Eq. \ref{eq:elbo_pgradient} is:
here we are able to use the gradient of $$r$$ directly, but in Eq. \ref{eq:elbo_pgradient} we rely
on the gradient of $$q_{\phi}$$ in order to increase the likelihood of $$x_i$$ that make $$r$$ large.
This is the same we did in the [Policy Gradients post](/lectures/lecture5), where we
discussed why doing this leads to an high variance estimator.
The figure below shows the process that from $$x_i$$ gives us $$q_{\phi}(z \vert x_i)$$ and $$p_{\theta}(x_i \vert z)$$:

{% include end-row.html %}
{% include start-row.html %}

{% include figure.html url="/_ml/prob_modelling/variational_inference/icon.png" description="" %}

{% include annotation.html %}
Notice this is what **Variational Autoencoders** do, first network being the **Encoder** $$p(z \mid x)$$ and second network being the **Decoder** $$p(x \mid z)$$.
{% include end-row.html %}
{% include start-row.html %}

#### A more practical form of $$\mathcal{L_i}(p, q)$$

If we look at Eq. \ref{eq:amortized_elbo}, we observe that it can be written in terms of the
KL Divergence between $$q_{\phi}$$ and $$p(z)$$:

\begin{equation}
\mathcal{L_i}(p, q) = E_{z \sim q_{\phi}(z \vert x_i)}\Big[\log p_{\theta}(x_i \vert z)\Big]+
\overbrace{
E_{z \sim q_{\phi}(z \vert x_i)}\Big[\log p(z) \Big]
+\mathcal{H}\Big(q_{\phi}(z \vert x_i)\Big)
}^{-D_{KL}\Big(q_{\phi}(z \vert x_i) \vert\vert p(z) \Big)}
\end{equation}
In practical implementations is often better to group the last two terms under the KL Divergence
since we can compute it analytically -e.g.
[D. Kingma, M. Welling, Auto Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf)-,
and we can use the reparametrization trick only on the first term.

{% include end-row.html %}
{% include start-row.html %}

The **Policy Gradient** for $$\mathcal{L_i}(p, q)$$ with respect to $$\phi$$ then becomes:
\begin{equation}
\label{eq:elbo_pgradient_dkl}
\boxed{
\nabla_{\phi} \mathcal{L_i}(p, q) =
\frac{1}{M} \sum_{j=1}^M \nabla_{\phi}q_{\phi}(z_j \vert x_i)\log p_{\theta}(x_i \vert z_j)
-\nabla_{\phi}D_{KL}\Big(q_{\phi}(z \vert x_i) \vert \vert p(z)\Big)
}
\end{equation}

The **Reparametrized Gradient** then becomes (single sample estimate):
\begin{equation}
\label{eq:elbo_trick_gradient_dkl}
\boxed{
\nabla_{\phi} \mathcal{L_i}(p, q) = 
\nabla_{\phi} \log p_{\theta}(x_i \vert \mu_{\phi}(x_i) + \epsilon \sigma_{\phi}(x_i)) -
\nabla_{\phi} D_{KL}\Big(q_{\phi}(z \vert x_i) \vert\vert p(z)\Big)
}
\end{equation}

{% include annotation.html %}
Notice that in both Eq. $$\ref{eq:elbo_pgradient_dkl}$$ and Eq. $$\ref{eq:elbo_trick_gradient_dkl}$$,
the first term ensures $$p(x_i)$$ is large and the second ensures $$q_\phi(z \mid x_i)$$ is close to the desired distribution of $$z$$: $$p(z)$$.
{% include end-row.html %}
{% include start-row.html %}

### Policy Gradient Approach or Reparametrization Trick?

**Policy Gradient** (Eq. \ref{eq:elbo_pgradient_dkl}):
+ <span style="color:green">Can handle both discrete and continuous latent variables</span>.
+ <span style="color:red">High variance, requires multiple samples and smaller
learning rates</span>.

**Reparametrized Gradient** (Eq. \ref{eq:elbo_trick_gradient_dkl}):
+ <span style="color:green">Low variance (one sample is often enough)</span>.
+ <span style="color:green">Simple to implement</span>.
+ <span style="color:red">Can handle only continuous latent variables</span>.

{% include end-row.html %}
