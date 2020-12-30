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


{% include end-row.html %}