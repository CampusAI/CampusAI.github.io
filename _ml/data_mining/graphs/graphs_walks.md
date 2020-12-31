---
layout: article
title: "Graph Walks"
permalink: /ml/graphs_walks
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

Given a graph $$G = (V, E)$$ we can model a random walk as a Markov chain, where the probability of moving from node $$n_i$$ into a connected node is $$\frac{1}{k_i}$$.

This procedure is very similar to the one presented in the **importance centrality measure**, but instead of starting with a vector of ones, you start with a 1-hot encoding of the initial node.
Notice that for any connected non-bipartite bidirectional graph and any starting point, the random walk (probability of being in each node) will converge to a unique stationary distribution.
Which in fact, will be the degree of each node (normalized).
**Connected** because otherwise its not unique and **non-bipartite** because otherwise it oscillates.
If the graph is directed, the requirement is that it is **[strongly connected](https://en.wikipedia.org/wiki/Strongly_connected_component)** and **[aperiodic](https://en.wikipedia.org/wiki/Aperiodic_graph)**.

Definitions:

{% include end-row.html %}
{% include start-row.html %}

- **Adjacency matrix**: $$A_{ij} = 1_{n_i \textrm{ connceted with } n_j}$$
- **Degree diagonal**: $$D := \textrm{diag}(\frac{1}{k_i})$$
- **Random Walk Transition Matrix**: $$m := D A$$. Encodes the probability of going from one node to its neighbors.

{% include annotation.html %}
Notice:
  - Row-sum of $$A$$ gives us the number 1-hop walks between each node.
  - Row-sum of $$A^2$$ gives us the number 2-hop walks between each node.
  - Row-sum of $$A^3$$ gives us ...
{% include end-row.html %}
{% include start-row.html %}

So if you take a 1-hot encoding vector $$v = 1_i$$, and multiply it as: $$v A^k$$, you get the number of ways you can reach each node with $$k$$ steps starting from node $$n_i$$.

Similarly, if we multiply it as $$v M^k$$, you get the probability of being at each node after $$k$$ steps starting from node $$n_i$$.
Repeatedly applying this idea, we will arrive at a vector $$\pi$$, such that: $$\pi M = \pi$$
I.e. $$\pi$$ will be an eigenvector of eigenvalue 1.

- **DEF**: **Mixing time**: Time until a Markov chain is close to being stationary.

## Graph Spectrum

{% include end-row.html %}
{% include start-row.html %}

Since $$A$$ and $$M$$ are symmetric they diagonalize with real eigenvalues ([Spectral Theorem](https://en.wikipedia.org/wiki/Spectral_theorem)).

**DEF**: We define **eigengap** (aka spectral gap) as: $$\lambda_1 - \lambda_2$$. The difference between first two eigenvalues. ($$1 - \lambda_2$$ for $$M$$).

Notice: If the graph is disconnected, we'll have several vectors with eigenvalue 1. In general, the spectral gap will tell us how well connected are the main components of the graph.
- If its very **small**: there are distinct components.
- if its very **large**:  there is only one main big component.

{% include annotation.html %}
**Notice**:
- Biggest Eigenvalue of $$A$$ will be the average degree of the graph
- Biggest Eigenvalue of $$M$$ will be the average degree 1.

**Notice**: If the graph is $$d$$-regular, the vector of ones will be an eigenvector of eigenvalue $$d$$.
{% include end-row.html %}
{% include start-row.html %}

## Expander graphs

**DEF**: Given a graph $$G = (V, E)$$, we define its **expansion** as:

\begin{equation}
\alpha := \min_{S \subseteq V} \frac{c_s}{\min (n_s, n - n_s)}
\end{equation}

Where: $$n = \vert V \vert$$, $$n_s = \vert S \vert$$ and $$c_s$$ is the number of edges which connect $$S$$ and $$V$$.

**Expander graphs**: Are graphs with $$\alpha \geq 1$$.
- They are sparse, yet very well connected graphs: It is very difficult to disconnect a large number of nodes.
- Thus they have a **large eigengap**: There is a single connected component, no different communities. At the exrteme, if gap is 0, the graph is already disconnected, so "effort" to disconnect it is 0.
- If d-regular, after a random walk of length $$O(\log (N))$$ the ending node distribution is uniform over all graph nodes.
- **Fast mixing time**: (rapid convergence of a random walk) $$O(\frac{\log N}{1 - \lambda_2}) \simeq O(\log N)$$. If the graph presents communities the convergence is much slower: chance of changing community and keep walking there.


## Web search

{% include end-row.html %}
{% include start-row.html %}

### PageRank Voting formulation

**Idea**: Every page has a weight (or rank) and it distributes it evenly among all the outgoing links. So each page importance is the sum of the votes on its in-links. Notice the similarity with **random walks** and the **importance centrality**.

- How to compute it? Using power iteration.

**Problems:**
- WWW is not strongly connected: It has sinks
- WWW is not aperiodic: It has loops

**Google solution:**
- Convert the graph into a weighted graph connecting weekly all nodes with each other and strongly connecting the real connections. This introduces a **teleportation component**, with a small probability a random walker might go to any node in the graph.

**More Problems:**
- WWW has $$\geq 5$$ billion pages! If you use an 4b int for the rank, this matrix alone takes $$10^11$$ TB!

**More Google solutions**


{% include annotation.html %}
SE histroy in 3 points:

- **1994**: [Yahoo! Directory](https://en.wikipedia.org/wiki/Yahoo!_Directory) was the most popular way of surfing the web. They introduced a search bar to navigate their manually indexed www content. Nevertheless, this manual model could not scale at the rate the www would do.

- **1995**: [AltaVista](https://en.wikipedia.org/wiki/AltaVista) based its searches on information retrieval, processing the text from the websites. Nevertheless, people could include a lot popular terms in their content and trick the search engine, making it unusable.

- **1998**: [Google](https://en.wikipedia.org/wiki/PageRank) introduced the idea of measuring webpage importance not by its content but but what other pages say of it. Making the rank much harder to trick. They later adopted the idea of selling searching queries turning the SE industry into one of the most profitable in the world.
{% include end-row.html %}
{% include start-row.html %}


{% include end-row.html %}