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


{% include annotation.html %}
It is easy to see how space grows exponentially with the dimensions.

- **1-d**: Imagine your data lies in a line of 10 units of length. The "volume" of space is $$10^1$$.
- **2-d**: If the data is instead in a plane of 10 units by side. The "volume" of space is $$10^2$$.

In 3-d the volume would be $$10^3$$...
The volume is growing exponentially with the number of dimensions as data-points become more isolated.
{% include end-row.html %}
{% include start-row.html %}

## Algorithms

So, how can we cope with high-dim data?

- Quantify **relevance** of dimensions and (possibly) eliminate some. This is commonly used in supervised learning, where we pay more attention to the variables which are more highly correlated to the target.

- Explore **correlation between dimensions**. Often a set of dimensions can be explained by a single phenomena which can be explained by a single latent dimension.

<!-- 
Dimensionality reduction algorithms present 3 main components:

- A **model**: Composed by an `encoder` which projects the data into a lower-dim (embedded) space and a `decoder` which recovers the original data from the embedded space.

- A **criterion** to be optimized. In dim-reduction problems the criterion is usually least squares over the representation: $$min E_x \left[ \Vert x - dec(enc(x)) \Vert_2^2 \right]$$. Other criteria can be: embedded space variance maximization or making latent variables independent.

- An **optimizer** to fit the criterion. -->

<!-- Furthermore, we can classify them into different categories:

- Linear vs non-linear
- Continuous vs discrete model
- External (latent dimensionality is hardcoded by user) vs Integrated
- Layered vs stand-alone embedding: Layered algorithms allow us to append new dimensions once fitted, stand-alone ones need to discard the embedding and start again.
- Batch vs online
- Exact vs Approximate -->

### **SVD**: Singular Value Decomposition

{% include end-row.html %}
{% include start-row.html %}
**Matrix decomposition** is expressing a matrix as a product of other matrices which certify some conditions.
{% include annotation.html %}
Some famous decompositions not presented in this post are: [LU Decomposition](https://en.wikipedia.org/wiki/LU_decomposition), [Cholesky Decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition), [Polar Decomposition](https://en.wikipedia.org/wiki/Polar_decomposition)
{% include end-row.html %}
{% include start-row.html %}

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
- $$U \in \mathbb{R}^{m \times m}$$ is an [orthogonal matrix](https://en.wikipedia.org/wiki/Orthogonal_matrix) (its cols form an [orthonormal basis](https://en.wikipedia.org/wiki/Orthonormal_basis)).
<!-- $$U^T U = I$$, $$U$$ rows and cols are orthogonal to each other and $$U^T = U^{-1}$$. -->
- $$\Sigma \in \mathbb{R}^{m \times n}$$ is diagonal with elements: $$\space \sigma_1 \geq \sigma_2 \ge ... \ge \sigma_k \ge 0$$. Where $$k = min\{m, n\}$$.
- $$V \in \mathbb{R}^{n \times n}$$ is an [orthogonal matrix](https://en.wikipedia.org/wiki/Orthogonal_matrix) (its cols form an [orthonormal basis](https://en.wikipedia.org/wiki/Orthonormal_basis)).
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
\min_{B s.t. rank(B) \le k} \Vert A - B \Vert_2 = \Vert A - A_k\Vert_2 = \sigma_{k+1}
\end{equation}
</blockquote>

This is all SVD dim-reduction does: Keep the k biggest singular values of the SVD and discard the rest.
We are approximating $$A$$ by only using the information of the gray areas of the SVD in this image:

{% include figure.html url="/_ml/dim_reduction/SVD.png"
description="Figure 1: Matrix approximation by selecting the first k singular values (gray area). Instead of representing $A$ by all its $m \times n$ values we can approximate it with $m \times k + k + k \times n \ll m \times n$. Figure from http://ezcodesample.com/."
%}

#### Interpretation

Lets first think of $$A \in \mathbb{R}^{n \times m}$$ as a [linear map](https://en.wikipedia.org/wiki/Linear_map) between two [vector spaces](https://en.wikipedia.org/wiki/Vector_space):

\begin{equation}
A: \mathcal{V} \subseteq \mathbb{R}^m \rightarrow \mathcal{U} \subseteq \mathbb{R}^n
\end{equation}

In this case, SVD finds an orthonormal base $$\mathcal{B}_\mathcal{V} = \{v_1, ... v_m\}$$ in $$\mathcal{V}$$ and another one $$\mathcal{B}_\mathcal{U} = \{u_1, ... u_n\}$$ in $$\mathcal{U}$$ such that between those bases $$A$$ is diagonal ($$\Sigma$$).

For instance: If $$A \in \mathbb{R}^{3 \times 2}$$,
SVD pairs: $$v_1 \rightarrow u_1, v_2 \rightarrow u_2$$.
Such that **EVERY** point $$p = (p_1 , p_2)_{\mathcal{B}_\mathcal{V}}$$ gets mapped to $$q = (\sigma_1 \cdot p_1 , \sigma_2 \cdot p_2, 0)_{\mathcal{B}_\mathcal{U}}$$:

{% include figure.html url="/_ml/dim_reduction/svd_interpretation.png"
description="Figure 2: Representation of A mapping SVD. Left singular vectors in the 2-d space are mapped to right singular vectors in 3-d space. In this case $\sigma_1 \gg \sigma_2$. Figure by CampusAI."
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
- If $$m \le n$$, $$A$$ maps space $$\mathcal{V}$$ into a subspace of dimension m in $$\mathcal{U}$$.

In particular, the one which $$A$$ presents a larger dilation. 

- If $$m \ge n$$, $$A$$ fills the whole destination space $$\mathcal{U}$$.
Keeping only the first $$k$$ dimensions will project into the  $$k$$-dim subspace of larger dilation.

{% include end-row.html %}
{% include start-row.html %}

##### Matrix as a transformation or matrix as data?

You now might think something like: _Wait, what if my matrix is not a transformation but just a table with data?_

{% include end-row.html %}
{% include start-row.html %}

Is it really _"just a table"_ though?
The [2006 Netflix prize](https://en.wikipedia.org/wiki/Netflix_Prize) is a great example to see the connection between matrices as data tables and matrices as transformations.
Netflix provided a dataset of users vs movie ratings as:

{% include annotation.html %}
I oversimplified this section for readers to just get the general idea, if interested in this topic read this great [post](https://jeremykun.com/2016/04/18/singular-value-decomposition-part-1-perspectives-on-linear-algebra/) on SVD interpretation.
{% include end-row.html %}
{% include start-row.html %}

{% include figure.html url="/_ml/dim_reduction/A_as_table.png"
description="Figure 4: Movie ratings examples. Figure from jeremykun.com."
%}

The prize was 1.000.000$ for team which achieved at least 10% less RMSE error than Cinematch (Netflix's recommender system used at that time).
And yeah, you guessed it: the winners used SVD (otherwise I wouldn't be talking about this duh).

So, where is the connection?
We can also understand this data matrix as a transformation from the user-space ($$\mathcal{V}$$) to the movie-space ($$\mathcal{U}$$):
If I'm 70% like Aisha 25% like Bob and 5% like Chandrika  my movie ratings will be a linear combination of Aisha's, Bob's and Chandrika weighted by my similarity to each one of them.
If do the dot product of the matrix to my similarity vector $$(0.7, 0.25, 0.05)$$ I get my most likely movie ratings (the matrix is now a transformation).

{% include end-row.html %}
{% include start-row.html %}

SVD finds a base in the user-space which is "aligned" to a base in the movie-space.
Each **right singular vector** $$v_i$$ encodes the person archetype who "just cares" for an associated archetype movie represented by a **left singular vector** $$u_i$$ with a strength of $$\sigma_i$$.

We can express each person as a combination of archetype people, which will like their associated archetype movies.
Moreover, we can pick $$k$$ to have as many archetypes as we want to cluster people's preferences.  

{% include annotation.html %}
For instance, you can think of an archetype person $$v_i$$ your *grandad* and an associated archetype movie $$u_i$$ *westerns*.
{% include end-row.html %}
{% include start-row.html %}

<!-- - _Ok, more or less... But we represent images as matrices... What about that?_ -->

##### Relation to Eigen Decomposition

[Eigen decomposition](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix) (aka spectral decomposition) is the decomposition of a **square** and **[diagonizable](https://en.wikipedia.org/wiki/Diagonalizable_matrix)** matrix into eigenvectors and eigenvalues:

\begin{equation}
A = Q \Lambda Q^{-1}
\end{equation}

Where $$Q = (v_1, ... v_n)$$ is an orthogonal matrix composed by the eigenvectors $$\{v_i\}_i$$ and $$\Lambda$$ is diagonal with the eigenvalues $$\{\lambda_i\}_i$$. Remember that an eigenvalue, eigenvector pair satisfy: $$A v_i = \lambda_i v_i$$, they represent the directions in which the transformation is a simple scaling.
For visual interpretation I recommend checking out this [video](https://www.youtube.com/watch?v=PFDu9oVAE-g&ab_channel=3Blue1Brown).

If you take a look on the [SVG theorem proof](http://gregorygundersen.com/blog/2018/12/20/svd-proof/) you'll see SVG is based on the Eigen decomposition of $$A A^T$$ and $$A^T A$$. In essence:

{% include end-row.html %}
{% include start-row.html %}

- $$\{\sigma_i \}_i$$: **singular values** fo $$A$$ are: $$\{ \sqrt{\lambda_i} \}_i$$ square root of eigenvalues of $$A^T A$$.
- **Right singular vectors** of $$A$$ (columns of $$V$$) are eigenvectors of $$A^T A$$.
- **Left singular vectors** of $$A$$ (columns of $$U$$) are eigenvectors of $$A A^T$$.

{% include annotation.html %}
If $$A \in \mathbb{R}^{n \times n}$$ is symmetric with non-negative eigenvalues, then eigenvalues and singular values coincide.
{% include end-row.html %}
{% include start-row.html %}

<!-- http://gregorygundersen.com/blog/2018/12/20/svd-proof/  -->

### **PCA**: Principal Component Analysis

Consider a dataset $$\mathcal{D}$$ of $$n$$ points in a high-dim space $$x_i \in \mathbb{R}^d$$. In a matrix form: $$X \in \mathbb{R}^{d \times n}$$.

{% include end-row.html %}
{% include start-row.html %}

Assumptions:
- For each data-point $$x_i \in \mathcal{D}$$ there exists a latent point in a lower-dim space $$z_i \in \mathbb{R}^k$$ which generates $$x_i$$.
- There exists a linear mapping (`decoder`) $$W \in \mathcal{R}^{d \times k}$$ s.t. $$x_i = W x_i \space \forall (x_i, x_i)$$
- $$W$$ has orthonormal columns (i.e. $$W^T W = I_{k \times k}$$, notice that usually $$W W^T \neq I_{d \times d}$$).


{% include annotation.html %}
Since $$W W^T = I_{k \times k}$$, we already have the `encoder`: $$z_i = W x_i$$.
{% include end-row.html %}
{% include start-row.html %}

So far we have a `decoder`:

\begin{equation}
    \textrm{dec}: \mathbb{R}^k \rightarrow \mathbb{R}^d \mid \textrm{dec(z)} = W z
\end{equation}

And an `encoder`:

\begin{equation}
    \textrm{enc}: \mathbb{R}^d \rightarrow \mathbb{R}^k \mid \textrm{enc(x)} = W^T x
\end{equation}

PCA aims to minimize the MSE between original data and the reconstruction: $$\min E_x \left[ \Vert t - dec(enc(x)) \Vert_2^2 \right]$$, which is:

\begin{equation}
\min_W E_x \left[ \Vert x - W W^T x \Vert_2^2 \right] = \min_W \underbrace{E_x \left[ x^T x \right]}_{\textrm{constant}} - E_x \left[ x^T W W^T x \right]
\end{equation}

Therefore we aim to minimize:

\begin{equation}
\min_W - E_x \left[ x^T W W^T x \right] =
\max_W E_x \left[ x^T W W^T x \right] =
\max_W \frac{1}{n} \sum_i^n x_i^T W W^T x_i =
\max_W \frac{1}{n} \text{tr} \left( X^T W W^T X \right)
\end{equation}

If we consider the SVD of $$X$$:

\begin{equation}
\text{tr} \left( X^T W W^T X \right) =
\text{tr} \left( V \Sigma U^T W W^T U \Sigma V \right)
\end{equation}

Which can be shown that the maximum is achieved by taking $$W = U_k$$.
 <!-- with value $$\sum_{i=1}^k \sigma_i^2$$. -->
SVD gives a nice connection between **explained variance** of the data and **reconstruction error** through singular values:
- First k ppal components variance: $$V_W = \sum_{i=1}^k \sigma_i^2$$
- Reconstruction error: $$E_W = \sum_{i=k+1}^d \sigma_i^2$$

We can use this idea to select an appropriate k to explain a certain percentage of data.
For instance, if we want our encoding to explain at least 85% of the data variance:

\begin{equation}
\min_k \mid \frac{V_W}{V_X} = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=d}^k \sigma_i^2} \ge 0.85
\end{equation}

#### Interpretation


[video](https://www.youtube.com/watch?v=_UVHneBUBW0&ab_channel=StatQuestwithJoshStarmer)
### **MDS**: Multi-Dimensional Scaling


{% include end-row.html %}