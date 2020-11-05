---
layout: article
title: "Dimensionality Reduction"
permalink: /ml/dim_reduction
content-origin: KTH DD2434, gregorygundersen.com
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

In this post we will first understand what are the main challenges of working with high-dim data.
Later, we will go through the most famous approaches to reduce data dimensionality while preserving as much information as possible.

## The curse of dimensionality

{% include end-row.html %}
{% include start-row.html %}

_The curse of dimensionality_ refers to the set of problems which arise when working in high-dim spaces.
These include:

- **Space volume grows exponentially** with the number of dimensions. Thus available data becomes very **sparse**. The volume is very large compared to the number of datapoints.

- Often in high-dim spaces many **dimensions are irrelevant** for the considered task.

- High-dim data **visualization is difficult**, making it hard to understand.

- Geometry in high-dim spaces is **not intuitive**.

So, how can we cope with high-dim data?

- Quantify **relevance** of dimensions and (possibly) eliminate some. This is commonly used in supervised learning, where we pay more attention to the variables which are more highly-correlated to the target.

- Explore **correlation between dimensions**. Often a set of dimensions can be explained by a single phenomena which can be explained by a single latent dimension.

{% include annotation.html %}
It is easy to see how space grows exponentially with the dimensions.

- **1-d**: Imagine your data lies in a line of 10 units of length. The "volume" of space is $$10^1$$.
- **2-d**: If the data is instead in a plane of 10 units by side. The "volume" of space is $$10^2$$.

In 3-d the volume would be $$10^3$$...
The volume is growing exponentially with the number of dimensions as data-points become more isolated.
{% include end-row.html %}
{% include start-row.html %}

## Algorithms

Dimensionality reduction algorithms present 3 main components:

- A **model**: Composed by an `encoder` which projects the data into a lower-dim (embedded) space and a `decoder` which recovers the original data from the embedded space.

- A **criterion** to be optimized. In dim-reduction problems the criterion is usually least squares over the representation: $$min E_x \left[ \Vert x - dec(enc(x)) \Vert_2^2 \right]$$. Other criteria can be: embedded space variance maximization or making latent variables independent.

- An **optimizer** to fit the criterion.

<!-- Furthermore, we can classify them into different categories:

- Linear vs non-linear
- Continuous vs discrete model
- External (latent dimensionality is hardcoded by user) vs Integrated
- Layered vs stand-alone embedding: Layered algorithms allow us to append new dimensions once fitted, stand-alone ones need to discard the embedding and start again.
- Batch vs online
- Exact vs Approximate -->

### **SVD**: Singular Value Decomposition

**Matrix decomposition** is expressing a matrix as a product of other matrices which certify some conditions.
Before explaining the idea behind SVD dimensionality reduction, lets take a look at the SVD theorem:

{% include end-row.html %}
{% include start-row.html %}

<blockquote markdown="1">
**SVD Theorem**:
Any matrix $$A \in \mathbb{R}^{m \times n}$$ can be factorized into:

\begin{equation}
A = U \Sigma V^T
\end{equation}

Where:
- $$U \in \mathbb{R}^{m \times m}$$ is an [orthogonal matrix](https://en.wikipedia.org/wiki/Orthogonal_matrix).
<!-- $$U^T U = I$$, $$U$$ rows and cols are orthogonal to each other and $$U^T = U^{-1}$$. -->
- $$\Sigma \in \mathbb{R}^{m \times n}$$ is diagonal with elements: $$\space \sigma_1 \geq \sigma_2 \ge ... \ge \sigma_k \ge 0$$. Where $$k = min\{m, n\}$$.
- $$V \in \mathbb{R}^{n \times n}$$ is an [orthogonal matrix](https://en.wikipedia.org/wiki/Orthogonal_matrix).
</blockquote>

{% include annotation.html %}
Geometrically SVD theorem is telling us that any linear transformation between vector spaces can be achieved by a **rotation** ($$V$$) in the src space, a **dilation + projection** ($$\Sigma$$) into the final space and another **rotation** ($$U$$) in the final space.

{% include end-row.html %}
{% include start-row.html %}

Naming goes:
- **Singular values**: $$\{\sigma_i \}_i$$
- **Right singular vectors**: Columns of $$V$$
- **Left singular vectors**: Columns of $$U$$

So, lets see how we can take advantage of this type of matrix decomposition:

{% include annotation.html %}
Notice that:
- $$A v_i = \sigma_i u_i$$.
- $$A^T u_i = \sigma_i v_i$$.
{% include end-row.html %}
{% include start-row.html %}

<blockquote markdown="1">
**Theorem**:
- Let $$A = U \Sigma V^T$$ be the SVD of $$A$$
- Let $$U_k = (u_1, ..., u_k)$$ be the first k cols of $$U$$
- Let $$V_k = (v_1, ..., v_k)$$ be the first k cols of $$V$$
- Let $$\Sigma_k = (\sigma_1, ..., \sigma_k)$$ be the k largest singular values

Then:
\begin{equation}
A_k = U_k \Sigma_k V_k^T
\end{equation}

Is the closest projection of $$A$$ into the space of matrices of rank-k w.r.t the **spectral norm**, and its distance to $$A$$ is $$\sigma_{k+1}$$:
\begin{equation}
\min_{B \ rank(N) \le k} \Vert A - B \Vert_2 = \Vert A - A_k\Vert_2 = \sigma_{k+1}
\end{equation}
</blockquote>

This is all SVD dim-reduction does: Keep the k biggest singular values of the SVD and discard the rest.
We are approximating $$A$$ by only using the information of the gray areas of the SVD in this image:

{% include figure.html url="/_ml/dim_reduction/SVD.png"
description="Figure 1: Matrix approximation by selecting the first k singular values (gray area). Instead of representing $A$ by all its $m \times n$ values we can approximate it with $m \times k + k + k \times n$ (way lesser). Figure from http://ezcodesample.com/."
%}

#### Interpretation

Lets first think of $$A \in \mathbf{R}^{n \times m}$$ as a [linear map](https://en.wikipedia.org/wiki/Linear_map) between two [vector spaces](https://en.wikipedia.org/wiki/Vector_space):

\begin{equation}
A: \mathcal{V} \subseteq \mathbf{R}^m \rightarrow \mathcal{U} \subseteq \mathbf{R}^n
\end{equation}

In this case, SVD finds an orthonormal base $$\mathcal{B}_\mathcal{V} = \{v_1, ... v_m\}$$ in $$\mathcal{V}$$ and another one $$\mathcal{B}_\mathcal{U} = \{u_1, ... u_n\}$$ in $$\mathcal{U}$$ such that between those bases $$A$$ is diagonal ($$\Sigma$$).

For instance: If $$A \in \mathbf{R}^{3 \times 2}$$,
SVD pairs: $$v_1 \rightarrow u_1, v_2 \rightarrow u_2$$.
Such that **EVERY** point $$p = (p_1 , p_2)_{\mathcal{B}_\mathcal{V}}$$ gets mapped to $$q = (\sigma_1 \cdot p_1 , \sigma_2 \cdot p_2, 0)_{\mathcal{B}_\mathcal{U}}$$:

{% include figure.html url="/_ml/dim_reduction/svd_interpretation.png"
description="Figure 2: Representation of A mapping SVD. Left singular vectors in the 2-d space are mapped to right singular vectors in 3-d space. In this case $\sigma_1 >> \sigma_2$. Figure by CampusAI."
%}

{% include end-row.html %}
{% include start-row.html %}

As we saw, SVD matrix approximation ignores the dimensions where the dilation is factor ($$\sigma_i$$) is close to $$0$$
In the image, $$\sigma_2$$ is much smaller than $$\sigma_1$$, if we remove the second dimension our mapping approximation will lie in a line.
The purple point will then be on the $$u_1$$ direction.

{% include figure.html url="/_ml/dim_reduction/svd_compression.png"
description="Figure 3: Effect of removing the second dimension. Notice that the purple point is relatively close to the `real` position. Figure by CampusAI."
%}

{% include annotation.html %}
In general:
- If $$m \le n$$, $$A$$ maps space $$\mathcal{V}$$ into a subspace of dimensional m in $$\mathcal{U}$$.

In particular, the one which $$A$$ presents a larger dilation. 

- If $$m \ge n$$, $$A$$ fills the whole destination space $$\mathcal{U}$$.
Keeping only the first $$k$$ dimensions will project into the  $$k$$-dim subspace of larger dilation.

{% include end-row.html %}
{% include start-row.html %}

You now might think something like:
- _Wait, what if my matrix is not a transformation but just a table with data?_

Is it really _"just a table"_ though? This [post](https://jeremykun.com/2016/04/18/singular-value-decomposition-part-1-perspectives-on-linear-algebra/) puts it very nicely.
In essence:
We can understand $$A \in \mathbf{R}^{n \times m}$$ as a collection of m $$n$$-dim points.


{% include figure.html url="/_ml/dim_reduction/A_as_table.png"
description="Figure 3: Effect of removing the second dimension. Notice that the purple point is relatively close to the `real` position. Figure from jeremykun.com."
%}

- _Ok, more or less... But we represent images as matrices... What about that?_


### Eigen Decomposition

- $$\{\sigma_i \}_i$$: singular values fo $$A$$ are: $$\{ \sqrt{\lambda_i} \}_i$$ square root of eigenvalues of $$A^T A$$.
- Right singular vectors of $$A$$ (columns of $$V$$) are eigenvectors of $$A^T A$$.
- Left singular vectors of $$A$$ (columns of $$U$$) are eigenvectors of $$A A^T$$.

<!-- http://gregorygundersen.com/blog/2018/12/20/svd-proof/  -->

### **PCA**: Principal Component Analysis

### **MDS**: Multi-Dimensional Scaling


{% include end-row.html %}