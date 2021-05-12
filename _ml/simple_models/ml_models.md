---
layout: article
title: "Simple ML models"
permalink: /ml/simple_models
content-origin: KTH DD2412
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

This post is more a memory-refresher list of simple ML models rather than a comprehensive explanation of anything in particular.

## Supervised Learning

### k-Nearest Neighbor (kNN)

{% include end-row.html %}
{% include start-row.html %}

**Solves**: Classification, Regression

**Method**: Estimate new point labels by considering only the labels of the $$k$$-closest points in the provided dataset.

**Pos/Cons:**
- <span style="color:green">Simplicity.</span>
- <span style="color:green">Only one meta-parameter.</span>
- <span style="color:red">If the dataset is very big, it can be slow to find closest point.</span>

**Characteristics:** `discriminative`, `non-parametric`

{% include annotation.html %}
{% include figure.html url="/_ml/simple_models/knn.png" description="It is a good idea to choose a $$k$$ different than the number of classes to avoid draws." width="50" zoom="1.75"%}
{% include end-row.html %}
{% include start-row.html %}

### Decision Trees

**Solves**: Classification, Regression

**Method**: Learn decision rules based on the data features.
These decisions are encoded by a tree structure where each node represents condition and its branches the possible outcomes.
The conditions are chosen iteratively granting a **higher information gain** (the split of data which results in the **lowest possible entropy entropy** state).

{% include end-row.html %}
{% include start-row.html %}

**Pos/Cons:**
- <span style="color:green">Intuitive and interpretable.</span>
- <span style="color:green">No need to pre-process the data.</span>
- <span style="color:green">No bias (no previous assumptions on the model are made)</span>
- <span style="color:green">Very effective, often used in winning solution in [Kaggle](https://www.kaggle.com/)</span>
- <span style="color:red">High variance (results entirely depend on training data).</span>

{% include annotation.html %}
This high-variance issue is often addressed with variance-reduction ensemble methods.
In fact, there is a technique specially tailored to decision trees called **random forest**:
A set of different trees are trained with different subsets of **data** (similar idea to **bagging**), and also different subsets of **features**.
{% include end-row.html %}
{% include start-row.html %}

**Characteristics:** `discriminative`

### Naive Bayes

**Solves**: Classification

{% include end-row.html %}
{% include start-row.html %}

**Method**: Given a dataset of labelled points in a $$d$$-dim space, a naive Bayes classifier approximates the probability of each class $$c_i$$ of a new point $$\vec{x} = (x^1, ..., x^d)$$ assuming independence on its features:

{% include annotation.html %}
**Naive** because it assumes independence between features.
{% include end-row.html %}
{% include start-row.html %}

\begin{equation}
p(\vec{x}) = p(x^1, ..., x^d) \simeq p(x^1) ... p(x^d)
\end{equation}

{% include end-row.html %}
{% include start-row.html %}

By applying the Bayes theorem:

\begin{equation}
p(c_i \mid \vec{x}) = \frac{p(\vec{x} \mid c_i) p(c_i)}{p(\vec{x})} \simeq \frac{p(x^1 \mid c_i) ... p(x^d \mid c_i) p(c_i)}{p(x^1)...p(x^d)}
\end{equation}

{% include annotation.html %}
Notice that we don't actually need to compute the **evidence** (denominator) as it only acts as a normalization factor of the probabilities.
It is constant throughout all classes.
{% include end-row.html %}
{% include start-row.html %}

{% include end-row.html %}
{% include start-row.html %}

We can then estimate:
-  $$p(x^j \mid c_i)$$ as the ratio of times $$x^j$$ appears with and without $$c_i$$ in the dataset.
-  $$p(c_i)$$ as the ratio of points classified as $$c^i$$ in the dataset.

**Pos/Cons:**
- <span style="color:green">Simplicity, easy implementation, speed...</span>
- <span style="color:green">Works well in high dimensions.</span>
- <span style="color:red">Features in real world are not independent.</span>

**Characteristics:** `generative`, `non-parametric`

{% include annotation.html %}
{% include figure.html url="/_ml/simple_models/naive_bayes.png" description="Example of how marginalization is done and independence is assumed." width="75" zoom="1.75"%}
{% include end-row.html %}
{% include start-row.html %}

### Support Vector Machines (SVMs)

{% include end-row.html %}
{% include start-row.html %}

**Solves**: Binary classification

**Method**: Learn the hyperplane parameters which "better" split the data.
It is framed as a **convex** optimization problem with the objective of finding the largest margin within the two classes of points.
The points from each group to optimize the hyperplane position are called **support vectors**.

- **Maximum Margin Classifier** attempts to find a "hard" boundary between the two data groups. It is however very susceptible to training outliers (high-variance problem).

- **Soft Margin** approach allows for the miss-classification of points, it is more biased but better performing.
Often cross-validation is used to select the support vectors which yield better results. Notice this is just a way to **regularize** the SVM to achieve better **generality**.

<blockquote markdown="1">
Schematically, SVM has this form:

\begin{equation}
\vec{x} \rightarrow  \text{Linear function} < \vec{x}, \vec{w} > \rightarrow
\begin{cases}
if \geq 1 \rightarrow \text{class 1} \newline
if \leq -1 \rightarrow  \text{class -1}
\end{cases}
\end{equation}
</blockquote>

{% include annotation.html %}
{% include figure.html url="/_ml/simple_models/svm.png" description="2-dim SVM." width="50" zoom="1.75"%}
{% include end-row.html %}
{% include start-row.html %}


##### Kernel
{% include end-row.html %}
{% include start-row.html %}
More often than not, the data is not linearly-separable.
We can apply a non-linear transformation into a higher-dimensional space and perform the linear separation there.
We call this non-linear transformation **kernel**.

**Pos/Cons:**
- <span style="color:green">Works well on high-dim spaces.</span>
- <span style="color:green">Works well with small datasets.</span>
- <span style="color:red">Need to choose a kernel.</span>

{% include annotation.html %}
**Kernel trick**: One can frame the problem in such a way that the only information needed is the dot product between points.
For some kernels we can calculate the dot product of the transformation of two points without actually needing to transform them, which makes the overall computation much more efficient.
We already saw this idea in [dimensionality reduction algorithms](https://campusai.github.io/ds/dim_reduction_algos).
{% include end-row.html %}
{% include start-row.html %}

**Characteristics:** `discriminative`

### Logistic Regression
{% include end-row.html %}
{% include start-row.html %}

**Solves**: Binary classification

**Method**: Apply a linear transformation to the input (parametrized by $$w$$) followed by a sigmoid function $$f(x) = \frac{e^x}{e^x + 1}$$:

\begin{equation}
\hat p (\vec{x}) = \frac{e^{< \vec{w}, \vec{x} >}}{e^{< \vec{w}, \vec{x} >} + 1}
\end{equation}

It uses **maximum likelihood estimation (MLE)** to learn the parameters $$\vec{w}$$ using a **binary cross-entropy** loss.

<blockquote markdown="1">
Schematically, it is very similar to a **SVM**:

\begin{equation}
\vec{x} \rightarrow  \text{Linear function} < \vec{x}, \vec{w} > \rightarrow \text{SIGMOID} \rightarrow 
\begin{cases}
if \geq 0.5 \rightarrow \text{class 1} \newline
if \leq 0.5 \rightarrow  \text{class 0}
\end{cases}
\end{equation}
</blockquote>

{% include annotation.html %}
{% include figure.html url="/_ml/simple_models/log_reg.png" description="1-dim logistic regression." width="75" zoom="1.75"%}

We can also think of this as a **single-layer ANN** with a SIGMOID activation function.

{% include end-row.html %}
{% include start-row.html %}

**Pos/Cons:**
- <span style="color:green">Simplicity.</span>
- <span style="color:red">Susceptible to outliers.</span>
- <span style="color:red">Negatively affected by high correlations in predictors.</span>
- <span style="color:red">Limited to linear boundaries.</span>
- <span style="color:red">Generalized and easily outperformed by ANNs.</span>

**Characteristics:** `discriminative`


### Linear Regression
**Solves**: Regression

{% include end-row.html %}
{% include start-row.html %}
**Method**:
Let $$X \in \mathbb{R}^{D \times n}$$ be a matrix where we arranged the n point of our dataset column-wise and $$Y \in \mathbb{R}^{d \times n}$$ the values we want to linearly map.

We look for the matrix $$W \in \mathbb{R}^{d \times D}$$ such that:

\begin{equation}
Y = W X
\end{equation}

{% include annotation.html %}
You can add a bias term by letting the first column of $X$ be made of $1$'s.
{% include end-row.html %}
{% include start-row.html %}

{% include end-row.html %}
{% include start-row.html %}

This leaves us with an overdetermined system of equations whose solution can be approximated using the **least squares method** (which minimizes the **mean squared error** loss):

\begin{equation}
W = Y X^T (X X^T)^{-1} 
\end{equation}

<blockquote markdown="1">
Schematically:

\begin{equation}
\vec{x} \rightarrow  \text{Linear function} \space \space W \cdot \vec{x} \rightarrow \vec{y}
\end{equation}
</blockquote>

**Pos/Cons:**
- <span style="color:green">Works well with small datasets.</span>
- <span style="color:red">Linear.</span>
- <span style="color:red">Susceptible to outliers and high correlations in predictors.</span>

{% include annotation.html %}
{% include figure.html url="/_ml/simple_models/linear_regression.png" description="1-dim linear regression." width="75" zoom="1.75"%}
{% include end-row.html %}
{% include start-row.html %}

{% include end-row.html %}
{% include start-row.html %}

**Characteristics:** `discriminative`

## Unsupervised Learning

{% include annotation.html %}
Notice any discriminative model can become generative if you are willing to assume some prior distribution over the input. 
{% include end-row.html %}
{% include start-row.html %}

### Clustering

#### Hierarchical clustering

{% include end-row.html %}
{% include start-row.html %}

**Method**: Start by considering each point of your dataset a cluster. Iteratively merge the closest cluster until you have as many clusters as you want.

**Pos/Cons:**
- <span style="color:green">Simple</span>
- <span style="color:red">Number of clusters need to be pre-set</span>
- <span style="color:red">Isotropic: Only works for spherical clusters</span>
- <span style="color:red">Optimal not guaranteed</span>
- <span style="color:red">Slow</span>

{% include annotation.html %}
{% include figure.html url="/_ml/simple_models/hier_clust.png" description="Hierarchical clustering idea." width="50" zoom="1.75"%}
{% include end-row.html %}
{% include start-row.html %}

#### k-means

{% include end-row.html %}
{% include start-row.html %}

**Method**: Randomly place $$k$$ centroids and iteratively repeat these two steps until convergence:
1. Assign each point of the dataset to the closest centroid.
2. Move the centroid to the center of mass of its assigned points.

**Pos/Cons:**
- <span style="color:green">Simple</span>
- <span style="color:red">Number of clusters need to be pre-set</span>
- <span style="color:red">Isotropic: Only works for spherical clusters</span>
- <span style="color:red">Optimal not guaranteed</span>

{% include annotation.html %}
{% include figure.html url="/_ml/variational_inference/k-means.gif" description="k-means visualized." width="50" zoom="1.75"%}
{% include end-row.html %}
{% include start-row.html %}

#### Expectation Maximization

{% include end-row.html %}
{% include start-row.html %}

See our post on [EM and VI](/ml/variational_inference)

<!-- {% include annotation.html %}
{% include figure.html url="/_ml/variational_inference/EM.gif" description="EM visualized" width="50" zoom="1.75"%}
{% include end-row.html %}
{% include start-row.html %} -->

#### Spectral clustering

**Method**: The algorithm consists of two steps:
1. Lower dimensionality of the data (using the [spectral decomposition of the similarity matrix](/ds/dim_reduction_algos))
2. Perform any simple clustering method (e.g. hierarchical or k-means) in the embedded space.

### Dimensionality Reduction

See our posts on [dimensionality reduction basics](/ds/dim_reduction_basics) and [dimensionality reduction algorithms](/ds/dim_reduction_algos).

### Generative models

See our post on [generative models](/ml/generative_models)

{% include end-row.html %}
