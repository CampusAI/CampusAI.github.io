---
layout: article
title: "Frequent Itemsets"
permalink: /ds/frequent_itemsets
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

## The Market-Basket Model

{% include end-row.html %}
{% include start-row.html %}

Say you have a shop selling a huge set of **items** $$\{i_1, ..., i_m \}$$ and have data of millions of **baskets**: small sets of items bought together by customers $$b_i \subseteq \{i_1, ..., i_m \}$$.

<!-- 
{% include annotation.html %}
For instance a basket of 2 items which are frequently bought together are soda and chips.
{% include end-row.html %}
{% include start-row.html %}
-->
We define the **support** of an itemset $$I$$ as the number of times this itemset is a subset of a basket:

\begin{equation}
\textrm{support} (I) = \vert \textrm{baskets containing all elems in} I \vert
\end{equation}

The simplest thing we can ask is: _"What are the most frequent itemsets?"_
Usually we pick a **support threshold** $$s$$, and all itemsets with support higher than that are considered frequent.

Sometimes we work with the **relative support**:

\begin{equation}
\textrm{rel_support} (I) = \frac{\textrm{support} (I)}{\vert \textrm{baskets} \vert}
\end{equation}

Another environment where this setup might be useful: documents as baskets and words as items.
In this case, if a set of items is frequent, we have that a set of words appears in a lot of documents.
This is specially interesting if the words are reasonably rare, which usually indicates that there is some connection between those concepts.

{% include annotation.html %}
Here are some ideas of things one can do with frequent itemsets information:
- Put frequent itemsets together in shelfs to further increase their joint sales.
- Run an offer on one of them while increasing the price of the others to trick people into spending more.

Note that you also need to consider some causality and this only makes sense in high-frequently bought items in offline stores.
Online stores can adapt to each customer and use different techniques.
{% include end-row.html %}
{% include start-row.html %}

## Association Rules

**Association rules** essentially say:
_If a basket contains the set of items $$X$$, then it will probably contain the set $$Y$$_.
Where $$X \cap Y = \emptyset$$
It is written as:

\begin{equation}
X \rightarrow Y
\end{equation}

Its **support** is defined as:

\begin{equation}
\textrm{support}(X \rightarrow Y) = \textrm{support}(X)
\end{equation}

Its **confidence** as:

\begin{equation}
\textrm{conf}(X \rightarrow Y) = \frac{\textrm{support}(X \cup Y)}{\textrm{support}(X)}
\end{equation}

And its **interest** as:

\begin{equation}
\textrm{interest}(X \rightarrow Y) = \textrm{conf}(X \rightarrow Y) - \textrm{rel_support} (Y)
\end{equation}

Cases:
- $$\textrm{interest}(X \rightarrow Y) \simeq 1$$: <span style="color:green">**Interesting!**</span> $$Y$$ appears only with $$X$$.
- $$\textrm{interest}(X \rightarrow Y) \simeq 0$$: <span style="color:red">**Not interesting**</span> Either there is low confidence or $$Y$$ appears everywhere.
- $$\textrm{interest}(X \rightarrow Y) \simeq -1$$:  <span style="color:green">**Interesting!**</span> $$Y$$ and $$X$$ never appear together. (e.g. Nobody buys both Cocacola & Pepsi)

### How to find association rules

Usually one wants to know all association rules with support $$\geq s$$, and confidence $$\geq c$$.
Lets see how can we do associations of the type $$\{i_1, ..., i_k\} \rightarrow j$$

<blockquote markdown="1">
**Algorithm:**
1. Find all sets with support of at least $$c \cdot s$$.
2. Find all sets with support of at least $$s$$. (Subset of the previous one)
3. If $$\textrm{support} (\{i_1, ... , i_k\}) \geq c \cdot s$$, see which subsets $$I_j = \{i_1, ... , i_k\} \setminus i_j$$ have support of at least $$s$$. We then know that $$I_j \rightarrow i_j$$ has support $$s$$ and confidence $$c$$.
</blockquote>

## Finding frequent pairs

**Computational model**: We assume the data is stored as an array of baskets, each of which containing an array of coded items.
Our main concerns will be to minimize the number of disk I/O's.
Notice our RAM is the critical limiting resource.

The hardest problem is to find an efficient way of getting **frequent pairs** (frequent itemsets of larger size are rare).
Furthermore, most techniques that work for 2 can be used to find larger itemsets.

{% include end-row.html %}
{% include start-row.html %}

<blockquote markdown="1">
**Naive Algorithm:**
1. Read data file once
2. For each basket, count its $$\binom{n}{2}$$ pairs by running two nested loops on its items.
</blockquote>

<span style="color:red">If using a triangular matrix to store data, this algorithm only works if $$\vert items \vert^2$$ does not exceed main memory.</span> 

While this approach works, its far from being efficient.
In next section, we explain the a-priori algorithm, which introduces a key idea to reduce memory and speed up the computation.

{% include annotation.html %}
Notice we can store the element count in a triangular matrix, or, more efficiently, use some [sparse matrix](https://en.wikipedia.org/wiki/Sparse_matrix) data structure.
{% include end-row.html %}
{% include start-row.html %}


## A priori algorithm

{% include end-row.html %}
{% include start-row.html %}

This algorithm efficiently finds association rules relying on a simple idea:
**The monotonicity of "frequent"**.
<!-- An itemset cannot be frequent unless all its subsets are frequent. -->
If a set of items has support $$s$$, all its subsets have suport at least $$s$$.

A-priori algorithm performs k-passes on the dataset (k being the size of the itemset to detect).

{% include annotation.html %}
For instance, for frequent pairs: if item $$i$$ does not appear in $$s$$ baskets, then no pair including i can appear in $$s$$ baskets.
{% include end-row.html %}
{% include start-row.html %}

<blockquote markdown="1">
**Algorithm:**

- **Pass 1:** Read baskets and count in main memory the occurrences of each item
- Keep only the frequent ones. (Support higher than s)

- **Pass 2:** Read baskets again and count only pairs both of which were found in pass 1
- Keep only the frequent ones. (Support higher than s)

- **Pass 3:** Read baskets again and count only triplets such that its pairs are frequent in pass 2.

- (...) Keep doing passes until no frequent itemset is found.
</blockquote>

There exist several developments on this algorithm.
Take a look at this [improvements of a-priori algorithm lecture](https://www.youtube.com/watch?v=AGAkNiQnbjY&ab_channel=MiningMassiveDatasets) and this [approximate algorithms one](https://www.youtube.com/watch?v=b-guME_xZiw&ab_channel=MiningMassiveDatasets).

{% include end-row.html %}