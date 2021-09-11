---
layout: page
title: Machine Learning
permalink: /ml/
---

<!-- This page contains explanations of diverse ML topics we found interesting. -->

<!-- ## Unsupervised Learning -->
## Basics

{% include paper-card.html
title="Probability Basics for ML"
subtitle="Definitions, MLE, Information theory, Statistical distances"
url="/ml/prob_modelling"%}

{% include paper-card.html
title="Probability Distributions"
subtitle="Bernoulli, Categorical, Binomial, Multinomial, Geometric, Poison, Uniform, Gaussian, Exponential, $\chi^2$, Gamma"
url="/ml/distributions"%}

{% include paper-card.html
title="ML Essentials"
subtitle="Basics, Regularization, Ensemble Methods, Error measures"
url="/ml/ml_concepts"%}

{% include paper-card.html
title="Simple ML models"
subtitle="kNN, Decision Trees, Naive Bayes, SVM, Logistic Regression, Linear Regression, Hierarchical Clustering, k-means, EM, Spectral Clustering"
url="/ml/simple_models"%}


## Generative Models

{% include paper-card.html
title="Why generative models?"
subtitle="Basics, Discriminative vs Generative, Use-cases, Types"
url="/ml/generative_models"%}

{% include paper-card.html
title="From Expectation Maximization to Variational Inference"
subtitle="Latent Variable Models, EM, VI, Amortized VI, Reparametrization Trick, Mean Field VI"
url="/ml/variational_inference"
star="no"%}

{% include paper-card.html
title="Autoregressive models (AR)"
subtitle="Basics, Simplification methods, Pro/Cons, Relevant Papers"
url="/ml/autoregressive_models"%}

{% include paper-card.html
title="Normalizing flows"
subtitle="Basics, Pro/Cons, Relevant Papers"
url="/ml/flow_models"%}

## Pay attention!

{% include paper-card.html
title="Attention Basics"
subtitle="Attention vs Memory"
url="/ml/attention"%}

## Annex

{% include paper-card.html title="Variational Inference Annex" subtitle="" url="/lectures/variational_inference_annex"   %}
