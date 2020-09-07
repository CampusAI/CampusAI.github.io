---
layout: paper
title: Out-of-Distribution Detection Using an Ensemble of Self Supervised Leave-out Classifiers
category: other
permalink: /papers/out_of_distribution_detection_ensemble
paper-author: Apoorv Vyas, Nataraj Jammalamadaka, Xia Zhu, Dipankar Das, Bharat Kaul, and Theodore L. Willke
post-author: Federico Taschin
paper-year: 2018
paper-link: https://arxiv.org/abs/1809.03576
---
<!--
Disclaimer and authorship:
This article is provided for free only for your personal informational and entertainment purposes. No commercial use of it is allowed.

Please note there might be mistakes. We would be grateful to receive (constructive) criticism if you spot any. You can reach us at: ai.campus.ai@gmail.com or directly open an issue on our github repo: https://github.com/CampusAI/CampusAI.github.io

If considering to use the text please cite the original author/s of the lecture/paper.
Furthermore, please acknowledge our work by adding a link to our website: https://campusai.github.io/ and citing our names: Oleguer Canal and Federico Taschin.
-->

Out-of-Distribution (OOD) Detection consists of being able to identify data that lies outside the
domain a Deep Learning system had been trained with. This can include recognizing when an image
does not belong to any of the known classes, detecting a sensor failure or a cyber attack.
In this paper, the authors propose an OOD detection algorithm that exploits an ensemble of
self-supervised classifiers trained by using a partition of the data as OOD data.


## Idea
Softmax outputs of classifiers are often seen as output probabilities for each class. However,
[Chang Guo et al., 2017](https://arxiv.org/abs/1706.04599) show that this is not often the case,
in particular when predicting inputs outside the range of training data. We cannot therefore
rely on softmax to estimate prediction uncertainty.

### Entropy based margin loss
This paper proporses a modified loss function for the classifiers training. Given a dataset of
in-disttribution samples $$(x_i \in X_{in}, y_i \in Y_{in})$$ and out-of-distribution samples
$$x_o \in X_{ood}$$, this loss term aims to maintain a margin of at least $$m$$ between the
average entropy of predictions for in and out distribution. Being $$F: x \rightarrow p$$ the
Neural Network function that maps an input $$x$$ to a distribution over classes, the loss is
given by

\begin{equation}
\mathcal{L} = -\frac{1}{\vert X_{in} \vert} \sum_{x_i \in X_{in}} \ln(F_{y_i}(x_i))
\underbrace{+\beta \max \Bigg( m + \frac{\sum_{x_i \in X_{in}}H(F(x_i))}{\vert X_{in}\vert}
-\frac{\sum_{x_o \in X_{ood}}H(F(x_o))}{\vert X_{ood}\vert},
0 \Bigg)}_{entropy-margin}
\end{equation}

**Notice:** $$F_{y_i}(x_i)$$ represents the predicted probability of class $$y_i$$ for input
$$x_i$$, while $$F(x)$$ represents the whole output distribution for input $$x$$. 

The function minimizes the **cross-entropy loss** given by the $$\ln(F_{y_i}(x_i))$$ terms.
The **entropy-margin** term has a maximum value of zero, and since we aim to\ minimize the total
loss $$\mathcal{L}$$, this term has to be maximized. In order to maximize it, we need the average
entropy of out-of-distribution data to be higher than that of in-distribution data by a margin
$$m$$. This has two effects:

- It **reduces** the average entropy for in-distribution predictions. This means that the softmax
  probabilities are pushed to be high for the ground truth class, and low for the others.
- It **increases** the average entropy for out-of-distribution data. This means that the softmax
  probabilities are pushed to be equal for all classes.

**Notice:** The marging $$m$$ is helpful in reducing overfitting. Without it the classifiers
could easily overfit in assigning high entropy to the out-of-distribution training data
partition at the cost of not generalizing to other unseen data. Keep in mind that this
out-of-distribution data is not really *out of distribution* as it is part of the dataset. 


### Training the ensemble
The training data $$X$$ is divided into $$K$$ mutually-exclusive partitions $$\{X_i\}$$, i.e.
no classes are contained by more than one partition. For each partition $$X_i$$, a classifier
$$F_i$$ is learned by using $$X_i$$ as out-of-distribution data and the rest of data partitions
as in-distribution data. The process is described in Algorithm 1.

{% include figure.html url="/_papers/ood_ensemble/algorithm1.png"
description="Learning K classifiers using a different partition as out-of-distribution data each"%}


## TODO Ablation studies
