---
layout: article
title: "Graph Clustering"
permalink: /ml/graphs_clustering
content-origin: mmds.org, KTH ID2222
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

**Community**: group of nodes with more edges within them than between them and other groups.

**Notation**:
- $$S$$: Set of nodes in the considered cluster
- $n_s = \vert S \vert$
- $$m_s$$: Number of edges within $$S$$: $$m_s = \vert \{ (u, v) : u, v \in S \} \vert$$
- $$c_s$$: Cut edges: Number of edges on the boundary of $$S$$: $$m_s = \vert \{ (u, v) : u \in S, v \notin S \} \vert$$
- $$f(S)$$ clustering quality of set $$S$$


## Connectivity Measures

{% include end-row.html %}
{% include start-row.html %}
We can measure the clustering quality looking at different metrics depending on what we are interested:

#### Internal Connectivity
This measures focus on what can we say about the cluster quality looking only at the nodes within the cluster.

{% include annotation.html %}
For a more complete study check out: [Defining and Evaluating Network Communities based on Ground-truth](https://arxiv.org/pdf/1205.6233.pdf).
Given ground-truth social communities: **conductance** and **triangular-partition-ratio** seem to give best performance.
Of course, depending on the application some might be more informative than others.
{% include end-row.html %}
{% include start-row.html %}

- **Edges inside**: <span style="color:red">Absolute value might not be very informative.</span>

\begin{equation}
f (S) = m_s
\end{equation}

- **Internal Connectivity**:

\begin{equation}
f (S) = \frac{m_s}{\binom{n_s}{2}}
\end{equation}

- **Average degree**:

\begin{equation}
f (S) = \frac{2 m_s}{n_s}
\end{equation}


- **Fraction over median degree (FOMD)**: Ratio of nodes in $$S$$ with higher degree than $$d_m$$, the median degree of the whole graph.

\begin{equation}
f (S) = \frac{\vert \{ u \in S : \vert \{ (u, v) \in E_S \} \vert > d_m \} \vert}{n_s}
\end{equation}

- **Triangle partition ratio**: Fraction of nodes in S that belong to a triangle.

#### External Connectivity

- **Expansion** (aka Cut Ratio): Number of edges that point outside per node of S.

\begin{equation}
f (S) = \frac{c_s}{n_s}
\end{equation}

Some people also define these measures as the fraction of existing edges out of all possible edges leaving $$S$$.

\begin{equation}
f (S) = \frac{c_s}{n_s (n - n_s)}
\end{equation}

#### Internal & External Connectivity

- **Conductance**:

\begin{equation}
f (S) = \frac{c_s}{2 m_s + c_s}
\end{equation}

- **Normalized cut**:

\begin{equation}
f (S) = \frac{c_s}{2 m_s + c_s} + \frac{c_s}{2 (m - m_s) + c_s}
\end{equation}

- **Average out degree fraction**: Average fraction of edges of nodes in $$S$$ that point outside $$S$$

\begin{equation}
f (S) = \frac{1}{n_s} \sum_{u \in S} \frac{\vert \{ (u, v) \in E : v \notin S \} \vert}{d_u}
\end{equation}

#### Based on network model

- **Modularity Q**: Measures the difference between the number of edges in $$S$$ and the expected number of edges in a random graph model of the same degree sequence. It answers the question: _"is this cluster more connected than the expected random graph?"_

For this, we define a **dense modularity matrix B**:

\begin{equation}
B_{ij} = A_{ij} - \frac{d_i d_j}{2 m}
\end{equation}

Where $$d_i, d_j$$ are the degrees of node $$i$$ and $$j$$ respectively.
Essentially we are subtracting the probability of the nodes being connected given the degrees.

Modularity is then given by the expression $$Q = \frac{1}{4 m} s^T B s$$, were is a vector indicating which nodes are considered inside the set.
<!-- Thus the vector which maximizes it will be the biggest eigenvalue eigenvector.
We can assign the cluster by the sign of the value of the node in the eigenvector-->

## Spectral Clustering

{% include end-row.html %}
{% include start-row.html %}

We will work with the adjacency (or affinity) matrix $$A$$ and the diagonal matrix:

\begin{equation}
D = 
\begin{bmatrix}
\sum_j a_{1j} & 0 & \cdots & 0 \newline
0 & \sum_j a_{2j} & \cdots & 0 \newline
\vdots & \vdots & \space & \vdots \newline
0 & 0 & \space & \sum_j a_{nj} \newline
\end{bmatrix}
\end{equation}

**Def**: $$W(A, B)$$ Number of edges between $$A, B$$.

{% include annotation.html %}
Notice this is not unique to graphs:
you can apply the following algorithms to a set of points from which you define some similarity measure between them and then define an affinity matrix $$A$$ (make sure its symmetric so its positive semi-definite: all eigenvals are real non-negative).
Sometimes using the whole dense matrix might be too costly $$O(n^2)$$, we can sparse it by thresholding or keeping knn.e might be more informative than others.
{% include end-row.html %}
{% include start-row.html %}

We can encode a cluster as a vector $$s = (1, 0, 0, 1, \cdots 0)$$ with a value of $$1$$ at position $$i$$ if node $$n_i$$ is in the cluster.
Since a node can only be in a single cluster we will have that for any pair of sets $$s_1, s_2$$: $$s_1^T s_2 = 0$$.

Then we have that $$A s$$ is the number of connections each node has with members of the set defined by $$s$$.
Thus, $$s_1^T A s_2$$ is the number of connections between $$s_1$$ and $$s_2$$ so:

\begin{equation}
W(s_1, s_2) = s_1^T A s_2
\end{equation}

Furthermore, notice that:

{% include end-row.html %}
{% include start-row.html %}

\begin{equation}
W(s_1, V) = s_1^T D s_1
\end{equation}
 
Where $$V$$ represents all the nodes in the graph.

{% include annotation.html %}
Equivalently: $$W(s_1, V) = s_1^T A 1$$
Where $$1$$ is a vector of ones.
{% include end-row.html %}
{% include start-row.html %}

Finally, we can express"

\begin{equation}
W(s_1, \bar s_1) =  W(s_1, V) -  W(s_1, s_1) = s_1 (D - A) s_1
\end{equation}

$$L = D - A$$ is known as the **Laplacian matrix**.
Notice all columns and rows add to 0.
Thus, the vector of 1's is an eigenvector of eigenvalue 0. 

{% include annotation.html %}
Notice the cardinality of the set defined by $$s$$ can be computed as: $$s^T s$$.
{% include end-row.html %}
{% include start-row.html %}

Now, we will try to find $$k$$ clusters in the graph that optimize some of the metrics we've presented previously.
Notice that this is a NP-hard problem so we will use much faster approximate methods.

### Maximizing average weight clustering (internal)

**Objective:** We want to maximize the number of edges within each cluster:

\begin{equation}
\max J_{av} = \max \sum_{i=1}^k \frac{W(S_i, S_i)}{\vert S_i \vert}
\end{equation}

So, if we substitute by the vector representation of the clusters we get:

\begin{equation}
J_{av} =
\sum_{i=1}^k \frac{s_i^T A s_i}{s_i^T s_i}
=
\sum_{i=1}^k \frac{s_i^T}{\Vert s_i \Vert} A \frac{s_i^T}{\Vert s_i \Vert}
=
\sum_{i=1}^k y_i A y_i
\end{equation}

Where we used that $$s_i^T s_i = \Vert s_i \Vert$$, and defined $$y_i := \frac{s_i}{\Vert s_i \Vert}$$ ($$s_i$$ normalized).
Thus we have a maximization problem with the constrain of $$\Vert y_i \Vert = 1$$, $$y_i^T y_j = \delta_{ij}$$ which we can solve with Lagrange Multipliers and get that: $$\max J_{av} = \sum_i^k \lambda_i$$. $$J_{av}$$ is the sum of the largest eigenvalues $$\lambda_i$$ of $$A$$.

Thus, the optimal solution will be the **$$k$$-largest eigenvalues** of $$A$$.

### Minimizing ratio cut (external)

**Objective:** We want to minimize the edges going out of each cluster:

\begin{equation}
\min J_{cut} = \min \sum_{i=1}^k \frac{W(S_i, \bar S_i)}{\vert S_i \vert}
\end{equation}

We have that:

\begin{equation}
J_{av} =
\sum_{i=1}^k \frac{s_i^T L s_i}{s_i^T s_i}
\end{equation}

Thus, we will look for the **$$k$$-smallest eigenvalues** of the Laplacian matrix $$L$$.

#### Rayleigh Theorem
{% include end-row.html %}
{% include start-row.html %}


But wait! We wanted solutions of $$s_i \in \{0, 1\}$$ and we have solutions of $$y_i \in \mathbb{R}$$!!

{% include annotation.html %}
This applies to bi-partitioning, further down the post I explain hoe to perform k-way spectral clustering.
{% include end-row.html %}
{% include start-row.html %}

With a bit of algebraic manipulation, we can see that:

\begin{equation}
\label{eq:rayleigh}
y^T L y = \sum_{(i, j) \in E} (y_i - y_j)^2
\end{equation}

And the minimum value of that expression is the 2^{nd} smallest eigenvalue $$\lambda_2$$, because the minimum eigenvalue is 0 which corresponds to the eigenvector of ones (all the elements in the same set).

Therefore, we know that $$y$$ will be a vector whose elements add up to 0 (since it has to be perpendicular to the previous eigenvector of ones).

We can think of $$y$$ as a vector which assigns positive values to the members of a cluster and negatives to the rest so that the connections between them are minimal as presented in eq \ref{eq:rayleigh}.

### Minimizing ratio cut (internal & external)


**Objective:** Minimize edges going out of each cluster normalized by the total number of edges connected to the cluster.

\begin{equation}
\min J_{con} = \min \sum_{i=1}^k \frac{W(S_i, \bar S_i)}{W(S_i, V)}
\end{equation}

Substituting:

\begin{equation}
J_{av} =
\sum_{i=1}^k \frac{s_i^T L s_i}{s_i^T D s_i}
\end{equation}

Remember that $$D$$ is Diagonal so $$D = D^\frac{1}{2} D^\frac{1}{2}$$ (also $$D^\frac{1}{2} D^{-\frac{1}{2}} = I$$).

Thus:

\begin{equation}
J_{av} =
\sum_{i=1}^k \frac{s_i^T D^\frac{1}{2} D^{-\frac{1}{2}} L D^{-\frac{1}{2}} D^\frac{1}{2} s_i}{(D^\frac{1}{2} s_i)^T (D^\frac{1}{2} s_i)}
=
\sum_{i=1}^k y_i^T D^{-\frac{1}{2}} L D^{-\frac{1}{2}} y_i
=
\sum_{i=1}^k y_i^T D^{-\frac{1}{2}} D D^{-\frac{1}{2}} y_i - \sum_{i=1}^k y_i^T D^{-\frac{1}{2}} A D^{-\frac{1}{2}} y_i
\end{equation}

Thus we need to find the vectors $$y_i$$ which maximize:

\begin{equation}
\sum_{i=1}^k y_i^T D^{-\frac{1}{2}} A D^{-\frac{1}{2}} y_i
\end{equation}

Again, we need to find the **$$k$$-largest eigenvectors** of $$D^{-\frac{1}{2}} A D^{-\frac{1}{2}}$$.

## k-Way Clustering

**Recursive bi-partitioning**: Recursively apply bi-partitioning in a hierarchical divisive manner. <span style="color:red">Inefficeint and unstable</span>

**Cluster multiple eigenvectors**: Build a reduced space from multiple eigenvectors. <span style="color:green">More commonly used in recent papers.</span> [paper](https://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf)

Essentially, you pick the k-eigenvectors which better split the graph (depending on what clustering approach you go for) and represent each node as a point in the projected space.
Naturally, this points will be better clustered together than the ones at the original space.
Then you can run some simple clustering algorithm on this projected space such as k-means.

**How do you select $$k$$?**
Most suitable number of clusters is given by the largest eigengap $$\max \Delta_k$$ where $$\Delta_k - \vert \lambda_k - \lambda_{k-1}\vert$$

## Overlapping oOmmunity Detection



{% include end-row.html %}