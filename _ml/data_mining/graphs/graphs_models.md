---
layout: article
title: "Graph Models"
permalink: /ml/graphs_models
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

In this section we'll focus on ways to generate graph with properties which resemble those by [social networks](http://erichorvitz.com/msn-paper.pdf).
In particular, we'll check for the following characteristics:

- **Degree distribution**: Power-law distribution.
- **Avg. path length**: $$O(\log N)$$
- **Avg. clustering coef**: $$\simeq 10\%$$
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

## Configuration Model

**Objective**: Given a degree sequence $$k_1, ..., k_N$$, build a graph which approximates it.

This algorithm gives a "close" solution:

<blockquote markdown="1">
**Algorithm**:
1. Assign each node a number of spokes corresponding to the degrees.
2. Randomly connect the spokes between nodes.
</blockquote>

{% include figure.html url="/_ml/data_mining/graphs/conf_model.png" description="Illustration of the graph construction. Image from mmd.org"%}

## Watts-Strogatz Model (Small-World Model)

**Problem:** Social networks present a very high clustering coefficient and yet a very low diameter. (Small-World situations)

<blockquote markdown="1">
**Algorithm**:
1. Construct a regular ring lattice of $$N$$ nodes connected to $$K$$ neighbors.
2. For each node, with probability $$p$$ re-wire a connecting edge.
</blockquote>

Applying this algorithm we obtain that for mid values of $$p$$, the graph has high clustering and low average path length:

{% include figure.html url="/_ml/data_mining/graphs/small_world.png" description="Illustration of the graph construction. Image from mmd.org"%}

{% include end-row.html %}