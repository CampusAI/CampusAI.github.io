---
layout: article
title: "Mining Data Streams"
permalink: /ml/mining_streams
content-origin: mmds.org
post-author: Federico Taschin
---
<!--
Disclaimer and authorship:
This article is provided for free only for your personal informational and entertainment purposes. No commercial use of it is allowed.

Please note there might be mistakes. We would be grateful to receive (constructive) criticism if you spot any. You can reach us at: ai.campus.ai@gmail.com or directly open an issue on our github repo: https://github.com/CampusAI/CampusAI.github.io

If considering to use the text please cite the original author/s of the lecture/paper.
Furthermore, please acknowledge our work by adding a link to our website: https://campusai.github.io/ and citing our names: Oleguer Canal and Federico Taschin.
-->
{% include start-row.html %}
In this post we discuss techniques for extracting information from Data Streams. When dealing with Data Streams we make the following assumptions:
- The data arrives in one or multiple streams and must be processed immediately.
- The size of the entire stream is too big to be stored in memory or is infinite.
- The distribution of the data may be non-stationary.

There are several questions (in this case called **queries**) that we may want to answer by extracting information from the stream(s).
These are generally divided in:
- **Sampling**
- **Queries on sliding windows**
- **Filtering**
- **Counting distinct elements**
- **Estimating moments**
- **Finding frequent items**

## Sampling
In this section we make the assumption that data comes in the form of tuples $$(k, x_1, x_2, ..., x_n)$$ where $$k \in K$$ is a key and $$\{x_i\}$$ are the attributes.
One such example may be data coming from a search engine in the form (*username*, query, date, location, ...) where *username* is key. Due to the assumptions above we cannot store the entire stream, but we may be interested in storing only a sample of it. Depending on whether we know the stream size in advance, we can do so with **Fixed proportion sampling** or **Fixed size sampling**.

### Fixed proportion sampling
Assume a memory constraint of $$M$$ tuples and a known stream size of $$N$$. The goal is to store the proportion $$p = M/N$$ of incoming tuples in such a way that each key has the same probability of being stored. 
We can then, for each tuple, hash its key with an hash function $$h: K \rightarrow \{1, ..., N\}$$ and store the tuple in memory if $$h(k) \le M$$.
If our hash function $$h$$ distributes the set of keys $$K$$ uniformly in $$\{1, ..., N\}$$ then this method approximately keeps a proportion $$p = M/N$$ of the incoming keys.
{% include end-row.html %}
