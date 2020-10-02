---
layout: paper
title: Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles
category: Uncertainty Estimation
permalink: /papers/uncertainty-estimation-deep-ensembles
paper-author: Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell
post-author: Oleguer Canal
paper-year: 2017
paper-link: https://arxiv.org/abs/1612.01474
---
<!--
Disclaimer and authorship:
This article is provided for free only for your personal informational and entertainment purposes. No commercial use of it is allowed.

Please note there might be mistakes. We would be grateful to receive (constructive) criticism if you spot any. You can reach us at: ai.campus.ai@gmail.com or directly open an issue on our github repo: https://github.com/CampusAI/CampusAI.github.io

If considering to use the text please cite the original author/s of the lecture/paper.
Furthermore, please acknowledge our work by adding a link to our website: https://campusai.github.io/ and citing our names: Oleguer Canal and Federico Taschin.
-->
{% include start-row.html %}

## Idea

Problems:
- Its hard to **quantify uncertainty** with classic ANNS as ground truth is not available. They usually produce **over-confident** results due to a lack of [calibration](https://towardsdatascience.com/neural-network-calibration-with-keras-76fb7c13a55).
- Bayesian NN can quantify it but are **slow** to compute, **prior-dependent**, and quality depends on the degree of **approximation** taken.

This paper combines multiple ideas to get an estimation of uncertainty:

### Use a 2-output network

One output for the **mean** $$\mu$$ and the other for the **variance** of the guess $$\sigma$$ as in [Estimating the mean and variance of the target probability distribution](https://ieeexplore.ieee.org/document/374138).
Samples are treated as from taken from a [heteroscedastic](https://en.wikipedia.org/wiki/Heteroscedasticity) Gaussian distribution.
They then use a Maximum Likelihood Estimation (MLE) on $$\mu$$, $$\sigma$$, minimizing the negative log-likelihood:

{% include end-row.html %}
{% include start-row.html %}

\begin{equation}
-\log p_\theta (y | x) =
\frac{\log \sigma_\theta^2 (x) }{2} +
\frac{(y - \mu_\theta (x))^2}{2 \sigma_\theta^2 (x)} + constant
\end{equation}

{% include annotation.html %}
Notice the trade-off between $$\mu$$ and $$\sigma$$. The optimizer can't just minimize $$\sigma$$ faster than $$\mu$$ since would make the second term grow.
{% include end-row.html %}
{% include start-row.html %}

### Use an ensemble of networks

It is known that an ensemble of models boosts **predictive accuracy**.
Bagging is often used to decrease variance while boosting to decrease bias. 
This research shows that it also improves **predictive uncertainty**.

Once trained, the ensemble is treated as a [mixture of Gaussians](https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95) of same weight.
Thus the final prediction is the mean of the mixture and the uncertainty is given by the [variance of the mixture](https://stats.stackexchange.com/questions/16608/what-is-the-variance-of-the-weighted-mixture-of-two-gaussians).
If using $$M$$ models with parameters: $$\theta_1 ... \theta_M$$:

\begin{equation}
\mu_\star (x) = \frac{1}{M} \sum_m \mu_{\theta_m} (x)
\end{equation}

\begin{equation}
\sigma_\star^2 (x) = \frac{1}{M} \sum_m \left( \mu_{\theta_m}^2 (x) + \sigma_{\theta_m}^2 (x) \right) - \mu_\star^2 (x)
\end{equation}

### Use adversarial training

When optimizing using adversarial training, a small perturbation on the input is created in the direction in which the network increases its loss.
This augmentation of the training set **smoothens the predictive distributions**.
While it had been used before to improve prediction accuracy, this paper shows that it also improves prediction uncertainty.

## Algorithm

The previously presented ideas can be combined in the following algorithm:

{% include figure.html url="/_papers/Uncertainty_Estimation_Deep_Ensembles/algorithm.png" description="Algorithm 1: Pseudocode of the proposed approach." zoom="1.5" %}

## Results
First they show on toy-regression examples the benefits of the 3 design choices explained above.
They then test their algorithm performance on: **regression**, **classification** and **uncertainty** estimation.

### Regression
They run their algorithm on multiple famous regression datasets (e.g. [Boston Housing](https://www.kaggle.com/c/boston-housing)).
- Their algorithm generally outperforms [PBP](https://arxiv.org/abs/1502.05336) and [MonteCarlo-Dropout](https://datascience.stackexchange.com/questions/44065/what-is-monte-carlo-dropout) algorithms using a [negative log-likelihood (NLL)](https://en.wikipedia.org/wiki/Likelihood_function#Log-likelihood) error metric.
- It performs slightly worse when using a [root mean squared error (RMSE)](https://en.wikipedia.org/wiki/Root-mean-square_deviation) metric for comparison. Authors claim this is due the fact that the loss they used whilst training was NLL.

### Classification
They test classification performance on the [MNIST](http://yann.lecun.com/exdb/mnist/) and [SVHN](http://ufldl.stanford.edu/housenumbers/) datasets.
- Adversarial training and a greater number of networks improve performance both for classification and [calibration](https://towardsdatascience.com/neural-network-calibration-with-keras-76fb7c13a55).
- The proposed method works better than [MonteCarlo-Dropout](https://datascience.stackexchange.com/questions/44065/what-is-monte-carlo-dropout).

### Uncertainty estimation

The paper evaluates uncertainty on **out-of distribution** examples (i.e unseen classes). 
To do so, they run the following experiment:

1. Train ensemble on MNIST train dataset.
2. Evaluate both on MNIST test (in-of distribution) and [NotMNIST](https://github.com/gholomia/notMNIST) (out-of distribution).
This evaluation consists in computing the entropy of network predictions.
The expectation is that out-of distribution examples present a uniform probability over outputs (with 10 outputs the entropy should be around $$H=-\sum_{i=0}^9 \frac{1}{10} \log_2 (\frac{1}{10}) \simeq 3.32$$).

They repeat the same experiment using the SVHN dataset for training and [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) for testing.
Results show that with a big enough ensemble their method is better calibrated than [MonteCarlo-Dropout](https://datascience.stackexchange.com/questions/44065/what-is-monte-carlo-dropout).
They can better estimate the uncertainty for out-of distribution inputs:
{% include end-row.html %}
{% include start-row.html %}
{% include figure.html url="/_papers/Uncertainty_Estimation_Deep_Ensembles/entropy.png" description="Figure 1: Entropy histograms for experimental results. Top row: in-of distribution test results. Bottom row: Out-of distribution test results. Notice that MC-dropout produces much over-confident results for inputs of an unseen distribution. In addition the number of networks in the ensemble plays a key role in uncertainty estimation." zoom="1.5" %}

{% include annotation.html %}
One can set a minimum confidence level (entropy score) to the model, otherwise output *"I don't know"*.

{% include end-row.html %}
{% include start-row.html %}

They show that their algorithm provides more reliable confidence estimates compared to MC-Dropout.

## Contribution

- New method to **estimate uncertainty** more reliable than contemporary ones.
- Little **meta-parameter** tuning.
- **Easy** to implement and apply to large scale distributed systems.

## Weaknesses

- Training an ensemble is **computationally expensive** and can be prohibiting in some cases.

{% include end-row.html %}
