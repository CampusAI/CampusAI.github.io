---
layout: article
title: "Similar Items"
permalink: /ml/similar_items
content-origin: KTH ID2222
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

In this post we will present a common approach to finding similar "documents" (e.g. articles, websites...) within a large set.
<!-- This can be used to detect related documents, or plagiarism.
If the "documents" are websites it can also be used as a way to detect mirrors (e.g. search engines do not want to show the same site twice). -->

## Problem definition

Say you have a corpus of documents and want to pair similar ones.
One idea would be to treat each document as a set of words and use some set similarity measure (e.g. Jaccard similarity) 

<blockquote markdown="1">
**Jaccard Similarity**: Measures the similarity between two sets by:

\begin{equation}
J_{SIM} (S_1, S_2) = \frac{\vert S_1 \cap S_2\vert}{\vert S_1 \cup S_2\vert}
\end{equation}
</blockquote>

<!-- **Problem**:
<span style="color:red">Word order not taken into account.</span>

Instead, a proposed approach is to follow these steps: -->
<!-- Shingling-> Minhashing -> LSH -->

### Shingling

{% include end-row.html %}
{% include start-row.html %}


The first step is to encode each document into a set of numbers easier to deal with.

<blockquote markdown="1">
**Algorithm:**
1. Split the given sequence in [k-grams](https://en.wikipedia.org/wiki/N-gram): Continuous subsequences of length k (aka k-shingles)
2. Hash each of the shingles into a 4B integer value and store them in a unique set.
</blockquote>

{% include annotation.html %}
**Example:** If text is `abcdab` and shingle length is $k=3$.
- The shingles will be: `abc`, `bcd`, `cda`, `dab`
- Which with some hash function they'll be mapped to for instance: `8`, `15`, `4`, `16`
- Thus this document is characterized by the set: `4`, `8`, `15`, `16`
{% include end-row.html %}
{% include start-row.html %}

Note that at this point we could pair-wise compare each document computing the **Jaccard similarity** between them.
<span style="color:red">Nevertheless, for big documents this is too expensive.</span>
$$O(N^2)$$ if done naively.
Thus, we will approximate the Jaccard similarity value using the Minhashing technique.

### Minhashing

{% include end-row.html %}
{% include start-row.html %}

Given the sets of hashed shingles $$S_1$$ and $$S_2$$, from documents: $$D_1$$ and $$D_2$$.
We are looking for a fast way to estimate their Jaccard similarity.

{% include annotation.html %}
Invented in 1997 and used in AltaVista search engine to detect website duplicates.
{% include end-row.html %}
{% include start-row.html %}

{% include end-row.html %}
{% include start-row.html %}
**Idea:** The probability of two sets having the same minimum is approximately their Jaccard similarity:

\begin{equation}
P(\min(S_1) = \min(S_2)) \simeq J_{SIM} (S_1, S_2)
\end{equation}

{% include annotation.html %}
Main intuition behind it is that the more common elements, the higher the likelihood of having the same minimum.
{% include end-row.html %}
{% include start-row.html %}

We can use this idea by repeatedly applying a hash function to our sets' elements and get an approximation of the similarity.

**DEF:** Given a hash function $$h$$ and a set $$S$$, we call **minhash** the value $$min_{e \in S} h(e)$$.

<blockquote markdown="1">
**Algorithm:**
Repeat m times:
1. Pick a hash function $$h_i$$
2. Hash each value of both sets
3. Check if minhashes match: $$\min(h_i(S_1)) == \min(h_i(S_2))$$

**Return**: Proportion of same minimum values.
</blockquote>

If comparing multiple sets, it is a good idea to store all minhashes to a matrix (each colum represents a set, each row a different hash function). This matrix is called **signature matrix**.

<span style="color:red">Still, this approximation might be too expensive for its accuracy.</span>
LHS further optimizes the computation.

### LSH (Locally-Sensitive Hashing)



{% include end-row.html %}
