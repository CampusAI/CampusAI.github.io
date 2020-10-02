---
layout: article
title: "Normalizing flow models"
permalink: /ml/flow_models
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

The main idea is to learn a deterministic [bijective](https://en.wikipedia.org/wiki/Bijection) (invertible) **mapping** from **easy distributions** (easy to sample and easy to evaluate density, e.g. Gaussian) to the **given data distribution** (more complex).

First we need to understand the **change of variables formula**: Given $$Z$$ and $$X$$ random variables related by a bijective (invertable) mapping $$f : \mathbb{R}^n \rightarrow \mathbb{R}^n$$ such that $$X = f(Z)$$ and $$Z = f^{-1}(X)$$ then:

{% include end-row.html %}
{% include start-row.html %}

\begin{equation}
p_X(x) = p_Z \left( f^{-1} (x) \right) \left|\det \left( \frac{\partial f^{-1} (x)}{\partial x} \right)\right|
\end{equation}

Were $$\frac{\partial f^{-1} (x)}{\partial x}$$ is the $$n \times n$$ Jacobian matrix of $$f^{-1}$$.
Notice that its determinant models the **local** change of volume of $$f^{-1}$$ at the evaluated point.

{% include annotation.html %}
"**Normalizing**" because the change of variables gives a normalized density after applying the transformations (achieved by multiplying with the Jacobian determinant). "**Flow**" because the invertible transformations can be composed with each other to create more complex invertible transformations: $$f = f_0 \circ ... \circ f_k$$.

{% include end-row.html %}
{% include start-row.html %}

{% include figure.html url="/_ml/generative_models/normalizing-flow.png" description="Figure 3: Normalizing flow steps example from 1D Gaussian to a more complex distribution. (Image from lilianweng.github.io)" zoom="1.0"%}

As you might have guessed, normalizing flow models parametrize this $$f$$ mapping function using an ANN $$(f_\theta)$$.
**This ANN**, however, needs to verify some specific architectural structures:

- <span style="color:red">Needs to be **deterministic**</span>
- <span style="color:red">I/O **dimensions** must be the **same** ($$f$$ has to be bijective)</span>
- <span style="color:red">Transformations must be **invertible**</span>
- <span style="color:red">Computation of the determinant of the Jacobian must be **efficient** and **differentiable**.</span>

Nevertheless they solve both previous approach problems:
- <span style="color:green">Present feature learning</span>.
- <span style="color:green">Present a tractable marginal likelihood</span>.

{% include end-row.html %}
{% include start-row.html %}

Most famous normalizing flow architectures ([NICE](https://arxiv.org/abs/1410.8516), [RealNVP](https://arxiv.org/abs/1605.08803), [Glow](https://arxiv.org/abs/1807.03039)) design layers whose Jacobian matrices are triangular or can be decomposed in triangular shape. These layers include variations of the **affine coupling layer**, **activation normalization layer** or **invertible 1x1 conv**.
Check out our [Glow paper post](/papers/glow) for further details on these layers.

{% include annotation.html %}
Some models combine the flows with the autoregressive idea creating **autoregressive flows**: Each dimension in the input is conditioned to all previous ones. Check out [MAF](https://arxiv.org/abs/1705.07057) and [IAF](https://arxiv.org/abs/1606.04934).

Similarly, flows can be applied to make VAEs latent space distribution more complex than Gaussian. Check out [f-VAES](rxiv.org/abs/1809.05861).

{% include end-row.html %}