---
layout: page
title: Machine Learning
permalink: /ml/
---

<!-- This page contains explanations of diverse ML topics we found interesting. -->

<!-- ## Unsupervised Learning -->

## Generative Models

{% include paper-card.html
title="Why generative models?"
subtitle="Basics, Discriminative vs Generative, Use-cases, Types"
url="/ml/generative_models"%}

{% include paper-card.html
title="From Expectation Maximization to Variational Inference"
subtitle="Latent Variable Models, EM, VI, Amortized VI, Reparametrization Trick, Mean Field VI"
url="/ml/variational_inference"%}

{% include paper-card.html
title="Autoregressive models (AR)"
subtitle="Basics, Simplification methods, Pro/Cons, Relevant Papers"
url="/ml/autoregressive_models"%}

{% include paper-card.html
title="Normalizing flows"
subtitle="Basics, Pro/Cons, Relevant Papers"
url="/ml/flow_models"%}

## Dimensionality reduction

{% include paper-card.html
title="Dim reduction basics"
subtitle="The curse of dimensionality, SVD"
url="/ml/dim_reduction_basics"%}

{% include paper-card.html
title="Dim reduction algorithms"
subtitle="PCA, KPCA, MDS, Isomap, AutoEncoders"
url="/ml/dim_reduction_algos"%}


## Massive Dataset Mining

{% include paper-card.html
title="Finding Similar Items: When $O(n^2)$ is not fast enough"
subtitle="Shingling, Minhashing, LSH"
url="/ml/similar_items"%}

{% include paper-card.html
title="Frequent Itemsets"
subtitle="Market-Basket Model, Association Rules, A-Priori Algorithm"
url="/ml/frequent_itemsets"%}

{% include paper-card.html
title="Graphs: Basics"
subtitle="Definitions, Paths, Centrality Measures"
url="/ml/graphs_basics"%}

{% include paper-card.html
title="Graphs: Models"
subtitle="Fixed edges, Erdos-Renyi, Preferential Attachment, Configuration, Watts-Strogatz"
url="/ml/graphs_models"%}


{% include paper-card.html
title="Graphs: Walks"
subtitle="?"
url="/ml/graphs_walks"%}

{% include paper-card.html
title="Graphs: Spectral Analysis"
subtitle="?"
url="/ml/graphs_walks"%}

## Annex

{% include paper-card.html
title="Probability Basics"
subtitle="Information theory, Statistical distances"
url="/ml/prob_modelling"%}

{% include paper-card.html title="Variational Inference Annex" subtitle="" url="/lectures/variational_inference_annex"   %}
