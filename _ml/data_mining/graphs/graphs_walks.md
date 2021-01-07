---
layout: article
title: "Walking on Graphs"
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

{% include end-row.html %}
{% include start-row.html %}
Given a graph $$G = (V, E)$$ we can model a random walk as a Markov chain, where the probability of moving from node $$n_i$$ into a connected node is $$\frac{1}{k_i}$$.

{% include annotation.html %}
Notice that this procedure is very similar to the one presented in the **importance centrality measure**.
Instead of starting with a vector of ones, we start with a 1-hot encoding of the initial node.
{% include end-row.html %}
{% include start-row.html %}

**Unique Convergence conditions:** For any **connected**, **non-bipartite**, and **bidirectional** graph, the random walk (probability of being in each node) converges to a unique stationary distribution.
Which in fact, will be the degree of each node (normalized).
- **Connected** because otherwise its not unique (depends on what cluster you start walking from)
- **Non-bipartite** because otherwise the random walk oscillates between the groups.

{% include annotation.html %}
In **directed** graphs, the requirements are **[strongly connectivity](https://en.wikipedia.org/wiki/Strongly_connected_component)** and **[aperiodicity](https://en.wikipedia.org/wiki/Aperiodic_graph)**.
{% include end-row.html %}
{% include start-row.html %}

Definitions:

{% include end-row.html %}
{% include start-row.html %}

- **Adjacency matrix**: $$A_{ij} = 1_{(n_i, n_j) \in E}$$
- **Degree diagonal**: $$D := \textrm{diag}(\frac{1}{k_i})$$
- **Random Walk Transition Matrix**: $$M := D A$$. Encodes the probability of going from one node to its neighbors. Essentially divides each row of $$A$$ by its sum (node degree).

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

**Mixing time**: Time until a Markov chain is close to being stationary.

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

**Expander graphs**: Are graphs with $$\alpha \geq 1$$ with the following properties:
- **Sparse**, yet very **well connected**: It is very difficult to disconnect a large number of nodes.
- **Large Eigengap**: There is a single connected component, no different communities. (At the extreme, if gap is 0, the graph is already disconnected, so "effort" to disconnect it is 0)
- If d-regular, after a random walk of length $$O(\log (N))$$ the ending node distribution is **uniform** over all nodes.
- **Fast mixing time**: (rapid convergence of a random walk) $$O(\frac{\log N}{1 - \lambda_2}) \simeq O(\log N)$$. If the graph presents communities the convergence is much slower: chance of changing community and keep walking there.

## Global information from walks

{% include end-row.html %}
{% include start-row.html %}

Imagine we implement a crawler on the Facebook graph and code it to move around checking properties of the nodes it visits.
This "random" walk will present several **biases**:
- More popular nodes will have a higher chance of getting visited.
- Age, nationality, activity, privacy awareness... are all correlated with node degree

{% include annotation.html %}
In a related note:
Ever wondered why friends have more friends than you do?
Check out the [friendship paradox](https://en.wikipedia.org/wiki/Friendship_paradox)
{% include end-row.html %}
{% include start-row.html %}


So, how can we get an unbiased sample?

### Metropolis-Hastings Random Walk (MHRW)

{% include end-row.html %}
{% include start-row.html %}

<blockquote markdown="1">
**Algorithm**:
Start at node $$n$$
Repeat:
- Sample neighbor node $$m$$
- If $$deg(m) \leq deg(n)$$: move to $$m$$
- If $$deg(m) > deg(n)$$: move to $$m$$ with probability $$p = \frac{deg(n)}{deg(m)}$$, else stay in $$n$$
</blockquote>

{% include annotation.html %}
When staying at the same node, you append it again on the visited node list.
This makes lower-degree nodes be better represented.
{% include end-row.html %}
{% include start-row.html %}

This gives you a list $$L$$ of nodes from which we can get less biased properties (we fix the more popular nodes will be visited more times problem).
Now, if we want to get some global property:

\begin{equation}
P(\textrm{property}) =
\frac{
\sum_{n \in S} I_{\textrm{property}} (n)
}
{\sum_{n \in S} 1}
\end{equation}


**Problem**: <span style="color:red">When visiting low-degree nodes with high degree connections, the algorithm can be stuck quite some time.</span>

### Re-Weighted Random Walk (RWRW)

This approach first performs a (biased) random walk and then applies the [Hansen-Hurwitz estimator](https://online.stat.psu.edu/stat506/lesson/3/3.2) to extrapolate to global conclusions:

\begin{equation}
P(\textrm{property}) =
\frac{
\sum_{n \in S} \frac{I_{\textrm{property}} (n)}{k_n}
}
{\sum_{n \in S} \frac{1}{k_n}}
\end{equation}



{% include end-row.html %}
