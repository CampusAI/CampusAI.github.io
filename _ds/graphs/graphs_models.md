---
layout: article
title: "Graph Models"
permalink: /ds/graphs_models
content-origin: mmds.org
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

In this section we'll focus on different ways of generating graphs and the properties that arise.
In particular, we will be interested in methods with properties which resemble those by [social networks](http://erichorvitz.com/msn-paper.pdf).
Mainly, the following characteristics:

- **Degree distribution**: [Power-law](https://en.wikipedia.org/wiki/Power_law) distribution.
- **Avg. path length**: $$O(\log N)$$
- **Avg. clustering coef.**: $$\simeq 10\%$$
- **Existence of a giant connected component**: Yes

## Fixed edges model: $$G(n, m)$$

<blockquote markdown="1">
**Algorithm**:
Start with $$n$$ isolated vertices and place m edges at random between them.
</blockquote>

## Erdos-Renyi random graph: $$G(n, p)$$

{% include end-row.html %}
{% include start-row.html %}
<blockquote markdown="1">
**Algorithm**:
Start with $$n$$ isolated vertices and connect each pair of nodes with a probability $$p$$.
</blockquote>

{% include annotation.html %}
Notice this family is bigger than the previous one.
{% include end-row.html %}
{% include start-row.html %}

Interestingly, **graph properties** such as diameter or probability of a giant component vs **$$p$$** present a threshold phenomena around $$p \simeq \frac{1}{N}$$.

The characteristics achieved by this graph family:

- **Degree distribution**: <span style="color:red">Gaussian distribution</span>
- **Avg. path length**: <span style="color:green">$$O(\log N)$$</span>
- **Avg. clustering coef**: <span style="color:red">$$\simeq 0\%$$</span>
- **Existence of a giant connected component**: <span style="color:green">Yes</span>

## Preferential attachment Model

{% include end-row.html %}
{% include start-row.html %}
<blockquote markdown="1">
**Algorithm**:
- Start with 2 connected nodes.

Repeat $$N-2$$ times:
- Add new node $$v$$
- Create a link between $$v$$ and one of the existing nodes with probability proportional to the degree
</blockquote>

{% include annotation.html %}
Notice this has the _"rich get richer"_ snowball effect.

The graphs created using this algorithm tend to look like stars.
{% include end-row.html %}
{% include start-row.html %}

Notice there will never be triangles since we do not add links to the rest of the graph, just between the new node.
The **Barbasi-Albert** model addresses this issue by allowing up to $$m$$ connections for each added node.

- **Degree distribution**: <span style="color:green">Power-law distribution</span>
- **Avg. path length**: <span style="color:green">$$O(\log N)$$</span>
- **Avg. clustering coef**: <span style="color:red">$$\simeq 0\%$$</span>
- **Existence of a giant connected component**: <span style="color:green">Yes</span>

## Configuration Model

{% include end-row.html %}
{% include start-row.html %}
**Objective**: Given a degree sequence $$k_1, ..., k_N$$, build a graph with an approximate sequence:

{% include annotation.html %}
Check out the [soft configuration model](https://en.wikipedia.org/wiki/Soft_configuration_model) for a version of this algorithm within the maximum-entropy framework.
{% include end-row.html %}
{% include start-row.html %}

<blockquote markdown="1">
**Algorithm**:
1. Assign each node a number of spokes corresponding to the degrees.
2. Randomly connect the spokes between nodes.
</blockquote>

{% include figure.html url="/_ds/graphs/conf_model.png" description="Illustration of the graph construction. Image from mmd.org"%}

- **Degree distribution**: <span style="color:green">Any distribution! :)</span>
- **Avg. path length**: <span style="color:green">$$O(\log N)$$</span>
- **Avg. clustering coef**: <span style="color:red">$$\simeq 0\%$$</span>
- **Existence of a giant connected component**: If the second moment minus twice the first moment of the inputed distribution is greater than zero.

## Watts-Strogatz Model (Small-World Model)

**Problem:** Social networks present a very high clustering coefficient and yet a very low diameter. (Small-World situations)

<blockquote markdown="1">
**Algorithm**:
1. Construct a regular ring lattice of $$N$$ nodes connected to $$K$$ neighbors.
2. For each node, with probability $$p$$ re-wire a connecting edge.
</blockquote>

{% include figure.html url="/_ds/graphs/small_world.png" description="Illustration of the graph construction. Image from mmd.org"%}

Usually for $$p \simeq \frac{K}{N}$$ we obtain the desired properties of:
- <span style="color:green">**High clustering coefficient**</span>: Achieved by the underlying lattice structure.
- <span style="color:green">**Low average path length**</span>: Achieved by the random re-wires.

Nevertheless, <span style="color:red">the degree distribution is **not power law**</span>, thus resulting in unrealistic distributions.

## Community-Affiliation Graph Model (AGM)

{% include end-row.html %}
{% include start-row.html %}
Given a set of nodes $$V$$ and communities $$C$$ we first assign each node to one or more communities:

<blockquote markdown="1">
**Algorithm**:

for each node:
- Assign one or more memberships to a given community with probability $$p_c$$
</blockquote>


{% include annotation.html %}
This is for graph with communities, check our [post on graph clustering](/ds/graphs_clustering) for more details
{% include end-row.html %}
{% include start-row.html %}

Thus, the model is uniquely defined by parameters: $$B (V, C, M, \{p_c\})$$.
Where $$M$$ are the memberships and $$p_c$$ the community probabilities.
Later, we create links within the communities:


<blockquote markdown="1">
**Algorithm**:

for each community $$A$$:

for each pair fo nodes within:
- Connect them with probability $$p_A$$
</blockquote>

Notice that the probability of two nodes being connected is:

{% include end-row.html %}
{% include start-row.html %}

\begin{equation}
p(u, v) = 1 - \prod_{c \in M_u \cap M_v} (1 - p_c)
\end{equation}

{% include annotation.html %}
- The more communities they have in common the larger the probability of being connected.
<!-- - The bigger the community (larger $$p_c$$), the smaller -->
- At the end, it is common to connect two random nodes with probability $$\epsilon$$.
{% include end-row.html %}
{% include start-row.html %}



{% include end-row.html %}