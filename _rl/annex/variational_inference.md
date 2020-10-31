---
layout: article
title: "Variational Inference"
permalink: /lectures/variational_inference_annex
post-author: Federico Taschin
---

{% include start-row.html %}
### The Evidence Lower Bound
In [Variational Inference](/ml/variational_inference) we showed that $$\ln p(x)$$ has a
lower bound that we called **Evidence Lower Bound**, which can be expressed in terms of some
latent variables $$z$$ and a distribution $$q(z)$$ -that in the post we use to approximate
$$p(z \vert x)$$.

By applying marginalisation of $$p(x)$$ over the latent variables and multiplying by
$$1 = q(z) / q(z)$$, we obtain:

\begin{align}
\ln p(x) =& \ln \int p(x \vert z)p(z) dz \\\\\\
          = & \ln \int \frac{q(z)}{q(z)} p(x \vert z)p(z) dz
\end{align}

We now realize that the integral can be expressed as an expected value over $$q(z)$$
\begin{equation}
\ln p(x) = \ln E_{z \sim q(z)} \left[ \frac{p(x \vert z)p(z)}{q(z)} \right]
\end{equation}

And, recalling **Jensen's Inequality**: $$\ln E[y] \ge E[\ln y]$$, we obtain
\begin{equation}
\ln p(x) \ge E_{z \sim q(z)}\left[ \ln \frac{p(x \vert z)p(z)}{q(z)} \right]
\end{equation}

Now, separating the terms using the properties of $$\ln$$, we obtain:

\begin{align}
\ln p(x) & \ge E_{z \sim q(z)} \Big[ \ln p(x \vert z) + \ln p(z) \Big] -
E_{z \sim q(z)} \Big[ q(z) \Big] \\\\\\
& = E_{z \sim q(z)}\Big[\ln p(x \vert z) + \ln p(z)\Big] + \mathcal{H}[q(z)]
\end{align}

As promised, we obtained the **Evidence Lower Bound** that in the post we denote with
$$\mathcal{L}(p, q)$$.

\begin{equation}
\label{eq:elbo}
\boxed{\mathcal{L}(p, q) = E_{z \sim q(z)}\Big[\ln p(x \vert z) + \ln p(x)\Big] +
\mathcal{H}[q(z)]}
\end{equation}


### Developing on the KL Divergence
In the post, we find another expression for $$p(x)$$, that includes the Evidence Lower Bound
we derived above, but this time the equality is exact. We obtain theis by developing on the
KL Divergence between $$q(z)$$ and $$p(z \vert x)$$. Remember that $$q(z)$$ is the distribution
we use to approximate $$p(z \vert x)$$ which we assume to be intractable.

The [KL Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) gives us
a measure of *how different* two distributions are. It has the property that it is always
greater than zero, and it is zero when $$q = p$$.

The KL Divergence is defined as
\begin{equation}
D_{KL}\Big(q(z) \vert\vert p(z \vert x) \Big) = E_{z \sim q(z)} \left[
\ln \frac{q(z)}{p(z \vert x)} \right]
\end{equation}

Substituting $$p(z \vert x) = \frac{p(x \vert z)p(z)}{p(x)}$$ we obtain

\begin{equation}
D_{KL}\Big(q(z) \vert\vert p(z \vert x) \Big) = E_{z \sim q(z)} \left[
\ln \frac{q(z)p(x)}{p(x \vert z)p(z)} \right]
\end{equation}

Which can be broken down and reordered as

\begin{align}
D_{KL}\Big(q(z) \vert\vert p(z \vert x)\Big) =&
\overbrace{-E_{z \sim q(z)}\Big[ \ln p(x \vert z) + \ln p(z)\Big]
-E_{z \sim q(z)}\Big[\ln q(z)\Big]}^{-\mathcal{L}(p, q)} +
\overbrace{E_{z \sim q(z)}\Big[\ln p(x)\Big]}^{\ln p(x)} \\\\\\
 =& -\mathcal{L}(p, q) - \ln p(x)
\end{align}

In which we used the fact that $$E_{z \sim q(z)}[\ln q(z)] = -\mathcal{H}[q(z)]$$ and therefore
we recognize in the first two terms the **Evidence Lower Bound** of Eq. \ref{eq:elbo}. Moreover,
since $$\ln p(x)$$ does not depend on $$z$$, the last term is just $$\ln p(x)$$.

Thus, we obtain the key result that will prove extremely useful in the
[post](/ml/variational_inference)

\begin{equation}
\boxed{\ln p(x) = D_{KL}\Big(q(z) \vert\vert p(z \vert x)\Big) + \mathcal{L}(p, q)}
\end{equation}

{% include end-row.html %}
