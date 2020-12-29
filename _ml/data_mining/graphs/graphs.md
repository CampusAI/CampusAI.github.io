---
layout: article
title: "Mining Graphs"
permalink: /ml/graphs
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

Network mining comes from the realization that sometimes it is more useful to look at the relation between entities than the entities themselves.
Some examples of big networks worth studying are: world-wide-web, social networks, roads, brain...

## Basics of Graph theory

A **graph** i a pair $$G = (V, E)$$, where $$V$$ is a set of objects called **nodes** (or **vertices**), and $$E$$ a set of **edges** (paired nodes).
Usually I will refer as $$N$$ to the total number of nodes, and $$E$$ the total number of edges.

There exist multiple types of graphs:
Undirected/Directed,
Unweighted/Weighted,
Bi/Tri/...partite,
Dense/Sparse (Dense: $$E = O(N^2)$$, being **complete** if $$E = \binom{N}{2}$$. In sparse graphs $$E = O(N log(N))$$).

{% include annotation.html %}
Google [PageRank algorithm](https://en.wikipedia.org/wiki/PageRank) is a good example of it:
the weight of each webpage is given by the number of links pointing to it, not by its content.

Another "fun" example is the way wikipedia checks for the reliability of its articles:
real articles have more coherent citations than fake ones. 
Reliable article citations both cite back the main article and between themselves.
Check out one of the most famous hoaxes which tricked experts into awarding a grant to study a made-up language: [Balboa Creole French](https://towardsdatascience.com/machine-learning-on-graphs-why-should-you-care-d9eb9a07a9d5)
{% include end-row.html %}
{% include start-row.html %}

<!-- 
#### Types of graphs

- **Undirected/Directed graph**: Edges equally connect two nodes/Edges point from one node to the other.

- **Unweighted/Weighted graph**: Weighted graphs assign weights to each edge. -->

<!-- - **Dense/Sparse graph**: $$E = O(N^2)$$. Being **complete** if $$E = \binom{N}{2}$$. In sparse graphs $$E = O(N log(N))$$. -->

<!-- - **Bipartite graph**: graph whose vertices can be divided into 2 disjoint sets U and V such that vertices only go between U and V and not within them. -->

<!-- - **Isomorphic graphs**: Same graph represented in different ways. -->

#### Definitions

<!-- - **Neighbors**: Two nodes who share an edge. -->

- **Degree** $$k_i$$: Number of adjacent nodes to node $$n_i$$.

- **Average Undirected Degree** $$\bar k = \frac{1}{N} \sum_{i=1}^N k_i = \frac{2 E}{N}$$

- **Directed Degree** $$k_i^{in}, k_i^{out}$$: Edges pointing into $$n_i$$, Edges pointing out of $$n_i$$ in a directed graph.

- **Average Directed Degree** $$\bar k = \frac{1}{N} \sum_{i=1}^N k_i^{in} + k_i^{out} = \frac{E}{N}$$

  - **Source**: Node such that $$k^{in} = 0$$
  - **Sink**: Node such that $$k^{out} = 0$$

{% include end-row.html %}
{% include start-row.html %}
- **Adjacency Matrix**: Matrix representing the connections between nodes: $$A_{ij} = 1_{\textrm{exists edge } n_i \textrm{ to } n_j}$$.


#### Paths

- **Path**: Sequence of nodes such that each consecutive pair is in edges.

{% include annotation.html %}
Notice you can get in and out degree of a node by respectively summing the row and column values of its associated adjacency matrix.
{% include end-row.html %}
{% include start-row.html %}

- **Distance** $$d_{ij}$$: Shortest path length between two nodes. Notice that if a graph is directed, distance is not symmetric. If two nodes are not connected the distance is usually taken as infinite.

- **Graph Diameter**: Can be measured in different ways:
  - **Longest distance**. (i.e. largest shortest path) Outliers can be an issue.
  - **Average path length**. (i.e. average length of all pairs of nodes) Disconnected nodes might be an issue. Usually we compute the average path length of a connected component as $$\hat h = \frac{\sum_{i \neq j} h_{ij}}{2 E_{max}}$$, where $$h_{ij}$$ is the distance from node $$n_i$$ to $$n_j$$. (this way we discard the infinite values).

<!-- - **Cycle**: Closed path with at least 3 edges. -->

#### Centrality Measures

- **Degree centrality**: $$c_D (i) := k_i$$. Takes the degree of a node as a measure of its centrality. <span style="color:red">Depends a lot on the size of the graph, just this number alone is not enough.</span>

- **Normalized degree centrality**: $$c_D^\star (i) := \frac{c_D (i)}{n - 1}$$. <span style="color:red">Does not consider graph topology.</span>

- **Closeness centrality**: $$c_C (i) = \frac{1}{\sum_j d(i, j)}$$. _"How close a node is to all others of the network"_.

- **Normalized Closeness centrality**: $$c_C (i) = \frac{n - 1}{\sum_j d_{ij}}$$. _"How close a node is to all others of the network"_. <span style="color:red">Only works for connected components.</span>

- **Harmonic centrality**: $$c_H (i) = \sum_j \frac{1}{d_{ij}}$$

{% include end-row.html %}
{% include start-row.html %}

- **Betweenness centrality**: _"Given a node, how many pairs of nodes have a shortest path through it"_. $$c_B (i) = \sum_{i \neq j \neq k} \frac{\sigma_jk (i)}{\sigma_jk}$$. Where $$\sigma_{jk} (i)$$ is the number of shortest paths from $$j$$ to $$k$$ through $$i$$, and $$\sigma_{jk}$$ the number of shortest paths from from $$j$$ to $$k$$.

- **Normalized Betweenness centrality**: $$c_B (i) = \sum_{i \neq j \neq k} \frac{\sigma_jk (i)}{\binom{n-1}{2}}$$

- **Edge Betweenness centrality**: Same but considering paths through a given edge, not node.

{% include annotation.html %}
A simple clustering technique using this metric is to iteratively remove the edge with highest betweenness centrality.
We thus hierarchically obtain the graph's clusters.
<span style="color:red">Notice it is very expensive though, we need to re-compute the edge betweenness centrality after each deletion.</span>
{% include end-row.html %}
{% include start-row.html %}

- **Importance centrality**: _"Importance of a node depends on the importance of its neighbors"_. If storing the importance of each node in $$v$$:

<blockquote markdown="1">
**Algorithm:**
1. Assign an importance of 1 to each node. ($$v=\textrm{ones}(N)$$)
2. $$v \leftarrow \frac{1}{\lambda} \sum_j A_{ij}v_j$$ ($$v$$ now holds the (normalized) degree of each node)
3. $$v \leftarrow \frac{1}{\lambda} \sum_j A_{ij}v_j$$ ($$v$$ now holds the (normalized) sum of degrees of each node's neighbors)
4. Repeat this operation until convergence
</blockquote>

Where $$\lambda$$ is some normalizing constant to ensure the iteration does not diverge.
Notice that by applying this algorithm we will reach an "importance" vector which satisfies: $$A v = \lambda v$$.
Therefore the stationary vector $$v$$ will be an eigenvector, in fact, the biggest eigenvalue eigenvector (aka **principal eigenvector**).

{% include annotation.html %}
This centrality is aka EigenVector centrality.

Similar centrality measures are [Katz centrality](https://en.wikipedia.org/wiki/Katz_centrality)
{% include end-row.html %}
{% include start-row.html %}

## Graph Models

## Graph Algorithms

{% include end-row.html %}