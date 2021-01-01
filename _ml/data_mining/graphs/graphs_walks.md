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
- **Random Walk Transition Matrix**: $$M := D A$$. Encodes the probability of going from one node to its neighbors.

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


## Ranking the WWW

### PageRank

{% include end-row.html %}
{% include start-row.html %}

**Idea**: Every page has a weight (or rank) and it distributes it evenly among all the outgoing links. So each page importance is the sum of the votes on its in-links. Notice the similarity with **random walks** and the **importance centrality**.

- How to compute it? Using power iteration.

#### Computational problems

- <span style="color:red">WWW is **not strongly connected**: It has sinks</span>
- <span style="color:red">WWW is **not aperiodic**: It has loops</span>

<span style="color:green">**Google solution:**</span> Convert the graph into a weighted graph connecting weekly all nodes with each other and strongly connecting the real connections. This introduces a **teleportation component**, with a small probability a random walker might go to any node in the graph. I.e. modify the matrix such that:

\begin{equation}
M_{\textrm{PageRank}} := 
\beta
\begin{bmatrix}
0 & \cdots & a_{1n} \newline
\vdots & \space & \vdots \newline
a_{n1} & \cdots & 0
\end{bmatrix}
+
(1 - \beta)
\begin{bmatrix}
0 & \cdots & \frac{1}{n} \newline
\vdots & \space & \vdots \newline
\frac{1}{n} & \cdots & 0
\end{bmatrix}
\end{equation}

- <span style="color:red">WWW has $$\geq 5$$ billion pages! If we now use a dense matrix of 4b ints for the rank, this matrix alone takes $$10^{11}$$ TB!</span>

<span style="color:green">**Google solution:**</span> Interpret teleportation as a "fixed tax", since it'll always be the same. So, instead of computing the rank vector using the dense PageRank Matrix $$r_{\textrm{new}} = r_{\textrm{old}} M_{\textrm{PageRank}}$$, we can directly use the **Random Walk Transition Matrix** $$r_{\textrm{new}} = r_{\textrm{old}} M + c$$. Where $$c := \frac{1-\beta}{N}$$ can be seen as a "tax".

{% include annotation.html %}
SE history in 3 points:

- **1994**: [Yahoo! Directory](https://en.wikipedia.org/wiki/Yahoo!_Directory) was the most popular way of surfing the web. They introduced a search bar to navigate their manually indexed www content. Nevertheless, this manual model could not scale at the rate the www would do.

- **1995**: [AltaVista](https://en.wikipedia.org/wiki/AltaVista) based its searches on information retrieval, processing the text from the websites. Nevertheless, people could include a lot popular terms in their content and trick the search engine, making it unusable.

- **1998**: [Google](https://en.wikipedia.org/wiki/PageRank) introduced the idea of measuring webpage importance not by its content but but what other pages say of it. Making the rank much harder to trick. They later adopted the idea of selling searching queries turning the SE industry into one of the most profitable in the world.
{% include end-row.html %}
{% include start-row.html %}

### Topic-Specific PageRank

Until now we've seen an algorithm to find the "generic popularity" of a page, but when searching, we are more interested in topic-specific popularity.

Instead of "teleporting" to any node we can define a **teleport set** $$S$$ (set of topic-related pages) which we would be able to randomly jump.
Applying the PageRank idea would then give us the rank of that particular page within that topic.
<!-- In addition, we could set different weights to pages within the teleport set. -->

Usually, we would pre-compute each **Topic-Specific PageRank** for each page and page-treated topic (arts, business, sports...).
Then, use the most relevant rank according to the user's query and context (e.g. locations, time, search history...)

### TrustRank

[**Spam Farming**](https://en.wikipedia.org/wiki/Link_farm): Of course there still are things people can do to boost their website rank:
- Add links to their website from accessible pages in the WWW (e.g. blogs, comments, reviews and other places the spammer can post).
- Create a set of "farm" pages pointing to the specific page, which also points back to them. 

While this farms could be attempted to be identified and blacklisted from PageRank, it might be a bit tricky to do so.

**TrustRank** idea: _"What if instead of trying to identify the bad guys, we identify the good guys (easier)"_.
Using the **Topic-Specific PageRank** idea, TrustRank implements a teleport set of only trusted sites, which can be selected in different ways:
- Pick top ranked sites of standard PageRank
- Pick sites from trusted domains (domains with controlled membership). E.g. .edu, .mil, .gov

If we apply **Topic-Specific PageRank** using this teleport set of reliable sources, we will get for each node a sense of "reliability".
We can then threshold them by some value and mark all those below the value as spam.

**Problems**:
- <span style="color:red">Working with **absolute value threshold** might be tricky since these scores depend on the size of the graph.</span>
- <span style="color:red">Some good pages might **naturally have low ranking**: maybe they are new or simply not very well connected.</span>

<span style="color:green">**Solution:**</span> Lets try to see what proportion of a page's PageRank comes from spam pages.

<blockquote markdown="1">
**Algorithm**:
For every page:
- $$r_p = \textrm{PageRank}(p)$$ (compute its PageRank)
- $$r_p^+ = \textrm{TrustedPageRank}(p)$$ (compute Topic-Specific PageRank with teleport set being trusted pages)
- $$r_p^- = r_p - r_p^+$$ (compute what fraction of the page comes from spam pages)
- $$p = \frac{r_p^-}{r_p}$$ (compute relative spam mass)
</blockquote>

Notice that $$p$$ is relative, so identifying spam pages becomes easier.

### Hubs-and-Authorities



### SimRank
{% include end-row.html %}
{% include start-row.html %}

<!-- [SimRank](https://en.wikipedia.org/wiki/SimRank) is an interesting application of **Topic-Specific PageRank** to find the most "similar" nodes from a given node. -->
Imagine we are given a graph and we are asked to measure "proximity" or "similarity" of all other nodes wrt to a given one $$u$$. Some ideas could be:
- **Shortest path length**: <span style="color:red">But does not consider the number of paths.</span>
- **Network flow**: <span style="color:red">Does not consider the length of the flow paths.</span>

{% include annotation.html %}
Check out the [SimRank paper](https://dl.acm.org/doi/10.1145/775047.775126) and this explanatory [video](https://www.youtube.com/watch?v=Y93J27otCWM).
{% include end-row.html %}
{% include start-row.html %}

{% include end-row.html %}
{% include start-row.html %}

Ideally, we would like a metric that considers both the number of connections and their quality (length, weights, intermediate node degrees...)
**[SimRank](https://en.wikipedia.org/wiki/SimRank)** algorithm proposes random walks from node $$u$$ with restarts (meaning all nodes point to the initial one with small probability).
The probability of being in each node doing a random walk is then taken as a "similarity" of that node to $$u$$.

{% include annotation.html %}
Essentially it is a Topic-Specific PageRank using as a **teleport set** $$S = \{ u \}$$ (the node we want to find similar items to).
{% include end-row.html %}
{% include start-row.html %}

It was initially proposed for k-partite graphs were k different types of entities point between each other (but not within).
For instance, imagine you have a 2-partite graph with a set of images connected with a set of tags:

{% include figure.html url="/_ml/data_mining/graphs/simrank.png" description="2-partite graph example of pictures and tags. Image from KTH ID2222"%}

Based on the connections between pictures and labels with SimRank we can get the most similar pictures to any given one: the ones more visited by random walks from $$u$$.

**Problem**: <span style="color:red">**Expensive**: you need to run the algorithm for each node you want to find similarities to.</span> Nevertheless it works well for sub-www problems.

{% include annotation.html %}
Interestingly, this was the algorithm used by [Pinterest](https://en.wikipedia.org/wiki/Pinterest) for content recommendation.
{% include end-row.html %}
{% include start-row.html %}


{% include end-row.html %}