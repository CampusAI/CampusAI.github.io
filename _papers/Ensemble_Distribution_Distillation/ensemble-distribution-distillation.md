---
layout: paper
title: Ensemble Distribution Distillation
category: Uncertainty Estimation
permalink: /papers/ensemble-distribution-distillation
paper-author: Andrey Malinin, Bruno Mlodozeniec, Mark Gales
post-author: Oleguer Canal
paper-year: 2019
paper-link: https://arxiv.org/abs/1905.00076
---
<!--
Disclaimer and authorship:
This article is provided for free only for your personal informational and entertainment purposes. No commercial use of it is allowed.

Please note there might be mistakes. We would be grateful to receive (constructive) criticism if you spot any. You can reach us at: ai.campus.ai@gmail.com or directly open an issue on our github repo: https://github.com/CampusAI/CampusAI.github.io

If considering to use the text please cite the original author/s of the lecture/paper.
Furthermore, please acknowledge our work by adding a link to our website: https://campusai.github.io/ and citing our names: Oleguer Canal and Federico Taschin.
-->

## Idea

Until recently ANNs were unable to provide reliable measures of their prediction **uncertainty**, which often suffer from **over-confidence**.
Model ensembles can yield improvements both in accuracy and [uncertainty estimations](/papers/uncertainty-estimation-deep-ensembles).
Their higher computational costs motivates their [distillation](https://arxiv.org/abs/1503.02531) into a single network.
While **accuracy** performance is usually **kept** after distillation, the information on the diversity of the ensemble (different types of **uncertainty**) is **lost**.

This paper presents a way to distill an ensemble of models while **maintaining the learned uncertainty**.

### Understanding uncertainty

- **Knowledge/epistemic uncertainty**: Uncertainty given by a lack of understanding of the **model**.
For instance: when an out-of distribution point or a point from a sparse region in the dataset is being assessed.
Can be fixed by training with more data from those regions.
- **Data/aleatoric uncertainty**: Irreducible uncertainty by the **data itself**.
For instance: because of the complexity, multi-modality or noise in the data. 
- **Total uncertainty**: Sum of knowledge and data uncertainties.

{% include figure.html url="/_papers/Ensemble_Distribution_Distillation/uncertainty.png" description="Figure 1: Types of uncertainty." %}

### Uncertainty from an ensemble

Consider an ensemble of **M models** in a **k-classes** classification task: $$\hat{\mathcal{M}} = \left\{\mathcal{M}_1, ..., \mathcal{M}_M \right\}$$ with parameters:
$$\hat{\theta} = \left\{\theta_1, ..., \theta_M \right\}$$.
Where $$\theta_m$$ can be seen as a sample from some underlying parameter distribution: $$\theta_m \sim p(\theta \mid \mathcal{D})$$.

Consider now the categorical distribution which the m-th model of the ensemble yields when a data-point $$x^\star$$ is evaluated: $$\pi_m = \left[ P(y=y_1 \mid x^\star , \theta_m), ..., P(y=y_1 \mid x^\star , \theta_m) \right]$$.
I.e. $$\mathcal{M}_m(x^\star) = \pi_m$$.
<!-- We can express $$\pi_m = f(x^\star; \theta_m)$$. -->
<!-- 
The model posteriors can be expressed as:
$$\left\{ P(y \mid x^\star, \theta_m) \right\}_{m=1}^M$$.
Or, equivalently, using only the categorical distributions:
$$\left\{ P(y \mid \pi_m) \right\}_{m=1}^M$$. -->

From these categorical distributions ($$\pi_m$$) we can see the sources of uncertainty of our ensemble.
$$\pi_m$$ can be understood as the [barycentric coordinates](https://en.wikipedia.org/wiki/Barycentric_coordinate_system) in a k-dimensional [simplex](https://en.wikipedia.org/wiki/Simplex).
Each corner of the simplex represents a different class in the classification task.
Geometrically, the probability of $$\pi$$ being a particular class is given by its distance to that particular corner.
For instance, in a 3-class classification problem, we can observe the following behaviors:

{% include figure.html url="/_papers/Ensemble_Distribution_Distillation/ensemble_uncertainty.png"
description="Figure 2: Representation of different uncertainties of an ensemble when evaluated on a test data-point in a 3-class task. In (a) all model predictions are close to the same corner (class), meaning all models agree. In (b) all  models agree but are uncertain about which class the data-point belongs to, this happens for in-distribution uncertain samples (data uncertainty). In (c) all models disagree, some being confident about different things, a symptom of an out-of distribution input." zoom="1.5" %}

**Notice**: The closer to the simplex center a $$\pi_m$$ distribution is, the higher the entropy $$\mathcal{H}$$ of that distribution (more uncertainty).

This interpretation helps us understand the following identities:

- **Expected data uncertainty**: $$E_{p(\theta \mid \mathcal{D})} \left[ \mathcal{H} \left[ P(y \mid x^\star, \theta )\right] \right]$$. That is: the **average of entropies** each model of the ensemble has.

- **Total uncertainty**: $$\mathcal{H} \left[ E_{p(\theta \mid \mathcal{D})} \left[ P(y \mid x^\star, \theta)\right] \right]$$ That is: the spread or "disagreement" between models in the ensemble. I.e. the **entropy of the average** of the predictions.

Therefore, from the ensemble we can infer the model uncertainty $$\mathcal{MI}$$ as:

\begin{equation}
\mathcal{MI} \left[ y, \theta \mid x^\star, \mathcal{D} \right] =
\mathcal{H} \left[ E_{p(\theta \mid \mathcal{D})} \left[ P(y \mid x^\star, \theta)\right] \right] - 
E_{p(\theta \mid \mathcal{D})} \left[ \mathcal{H} \left[ P(y \mid x^\star, \theta )\right] \right]
\end{equation}

### Ensemble distribution distillation

In order to maintain both the predictive accuracy and diversity of the ensemble the authors use
[prior networks](https://papers.nips.cc/paper/7936-predictive-uncertainty-estimation-via-prior-networks.pdf).
Prior networks: $$p(\pi \mid x; \phi)$$ model a **distribution over categorical output distributions** $$\left( \{\pi_m \}_m \right)$$.
The [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution) distribution is chosen for its tractable analytic properties (allows closed-form expressions):

\begin{equation}
p(\pi \mid x; \hat \phi) = Dir (\pi \mid \hat \alpha)
\end{equation}

Where $$\hat \alpha$$ is the **concentration parameters** vector: $$\hat \alpha_c > 0, \hat \alpha_0 = \sum_{c=1}^k \hat \alpha_c$$.
And $$\hat \phi$$ are the set of parameters which map each input data-point $$x$$ to its associated concentration parameters $$\hat \alpha$$: $$\hat \alpha = f (x; \hat \phi)$$.
This parameters can be fitted using [MLE](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) on a dataset with all input and output of each network of the ensemble:
$$\mathcal{D_e} = \left\{ x_i \pi_{i, 1:M} \right\}_{i=1}^N \sim \hat p (x \mid \pi)$$.
To do so, we simply minimize the following loss:

\begin{equation}
\mathcal{L} \left(\phi, \mathcal{D_e} \right) = - E_{p(x)}
\left[
E_{\hat p (\pi \mid x)}
\left[
\log p (\pi \mid x ; \phi)
\right]
\right]
\end{equation}

Which means (_if my interpretation is not mistaken_) that we get the parameters as:

\begin{equation}
\phi^\star = \arg \min_\phi - \sum_{(x_i, \pi_{i, j}) \in \mathcal{D_e}}
\log p(\pi_{i, j} \mid x_i, \phi)
\end{equation}

**Notice**: Often training output distributions will be very sharp on a corner of the simplex (as in figure 2.a).
Nevertheless, the initial parameters of the Dirichlet distribution are closer to the center of the simplex (assume unknown).
Training with this disparity can be challenging and authors introduce a **temperature annealing schedule**.
They "heat" or "move" first optimization steps to make the distributions more uncertain and then gradually decrease this temperature.

## Contribution

## Weaknesses

- Choosing a **Dirichlet distribution might be too limiting**: The ensemble outputs could follow a different distribution.