---
layout: article
title: "Probability Basics for ML"
permalink: /ml/prob_modelling
content-origin: KTH DD2412
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

This post includes basic probability concepts useful to have in mind when working with ML models.
Its main purpose is to be a memory refresher.
Thus, it is written as schematic notes avoiding formal definitions.

## Definitions

- **Random variable (rv)**: Mapping between **events** (or outcomes) and **real values** that ease probability calculus.
  
- **Independence**: We say events $$X$$, $$Y$$ are independent when the occurrence of $$X$$ doesn't affect $$Y$$: $$p(X \cap Y) = p(X) \cdot p(Y)$$

- **Probability distribution**: Function that provides occurrence probability of possible outcomes.
    - **PMF** (Probability Mass Function): Probability of each point in a discrete rv.
    - **PDF** (Probability Density Function): Likelihood of each point in a continuous rv. Or probability spread over area.

- **Expectation**: Weighted average of the distribution (aka center of mass).
  - In discrete rv: $$E [X] = \sum p(x) x$$ 
  - In continuous rv: $$E [X] = \int p(x) x $$ 

- **Variance**: Expectation of the squared deviation of a rv: $$V(X) = E[(X - E[X])^2]$$

- **Conditional probability**: $$p(X \mid Y) = \frac{p(X \cap Y)}{p(Y)}$$

- **Bayes Theorem**:

\begin{equation}
p(Y | X) = \frac{p(X | Y) p(Y)}{p(X)}
\end{equation}

Naming goes: **Posterior**:  $$P(y \mid x)$$. **Likelihood**:  $$P(x \mid y)$$. **Prior**: $$P(y)$$. **Evidence**: $$P(x)$$.


## Maximum likelihood estimation

{% include end-row.html %}
{% include start-row.html %}

MLE finds the parameters which better explain the data.
I.e. which achieve a higher likelihood on the distribution assumed (through "expert" knowledge or distribution tests):

\begin{equation}
\theta^\star = \arg \max_\theta (\mathcal{L} (\mathcal{D}, \theta))
\end{equation}

{% include annotation.html %}
In some cases (see [Deep Generative Models](https://campusai.github.io/ml/generative_models)) the assumed functional form of the likelihood is so complex that this optimization cannot be done analytically.
We then use other optimization techniques (s.a. gradient-based methods).
{% include end-row.html %}
{% include start-row.html %}




## Information Theory Basics

### Shannon Information
\begin{equation}
I_P(X) = - \log (P (X))
\end{equation}
**How surprising** an observed event is.
- If $$p(X) \simeq 1$$: $$X$$ is "$$0$$-surprising"
- If $$p(X) \simeq 0$$: $$X$$ is "$$\infty$$-surprising"

### Entropy
\begin{equation}
\mathcal{H} (P) = E_P [ I_P (X)]
\end{equation}
**How surprised you expect to be** by sampling from P.
It is the average of information content of a probability distribution.
- **High entropy** $$\rightarrow$$ **High uncertainty** (a lot of results are similarly likely)
- **Low entropy**  $$\rightarrow$$ more deterministic outcomes (very few results are very likely)

### Cross-Entropy
\begin{equation}
\mathcal{H} (P, Q) = E_P [ I_Q (X)]
\end{equation}
Information content of distribution Q weighted by distribution P.
- If something is very surprising to Q and very probable in P cross entropy will be high.
- The closer $$Q$$ is to $$P$$, the closer $$\mathcal{H} (P, Q)$$ will be to $$\mathcal{H} (P)$$.

## Statistical distance
[Generalized metric](https://en.wikipedia.org/wiki/Generalised_metric) to quantify dissimilarity between random variables, probability distributions or samples.
Some probability divergence metric examples are: [Kullback–Leibler](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence), [Jensen–Shannon](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) or [Bhattacharyya distance](https://en.wikipedia.org/wiki/Bhattacharyya_distance).

{% include figure.html url="/_ml/prob_modelling/prob_basics/distribution_space.svg" description="Figure 1: Representation of probabilistic modelling optimization. (Image by CampusAI)"%}

A lot of ML models attempt to minimize a statistical distance to find the best parameters of a parametrized distribution.
The ultimate goal is to have a model which behaves as close as possible to the "real" distribution.
Obviously this distribution is not known and we are only given some samples.
The basic approach is to choose a model which can capture a subspace of the probability distribution space.
Later, to optimize the model parameters to minimize the distance to the given distribution.

### Kullback–Leibler (KL) divergence
Often used as an approximation technique in Bayesian modelling when expressions are untractable.

\begin{equation}
D_{KL} (P || Q) = E_P [ I_Q (X) - I_P (X)]
\end{equation}

**How surprised** you get on average **observing Q** knowing that **events follow** distribution **P**.

Developing the expression we get:

{% include end-row.html %}
{% include start-row.html %}

\begin{equation}
D_{KL} (P || Q) = -
\sum P (x) \log (Q (x)) +
\sum P (x) \log (P (x)) = 
\mathcal{H} (P, Q) - \mathcal{H} (P)
\end{equation}

{% include annotation.html %}

The minimum achievable value is 0. This happens when $$P=Q$$, in this case $$\mathcal{H} (P, Q) = \mathcal{H} (P)$$

If P is a fixed distribution, then $$\mathcal{H} (P)$$ is constant, so minimizing KL-divergence is the same as minimizing Cross-Entropy. That is why cross entropy is often used as a loss when minimizing NN (P are labels, Q are predictions).

{% include end-row.html %}
{% include start-row.html %}

#### Asymmetry in KL divergence:

If $$P$$ is the "true" distribution and $$Q$$ our approximation we have 2 types of KL divergence:

- **Forward KL** (moment-matching): $$D_{KL} (P \mid \mid Q) = \sum P \log \frac{P}{Q}$$. Where P is small, it doesn't matter what Q is doing. E.g. if Q is Gaussian, it will only try to match the moments of P (mean and variance). **Forward KL** is what most ANNs optimize.

- **Reverse KL** (mode-seeking): $$D_{KL} (Q \mid \mid P) = \sum Q \log \frac{Q}{P}$$. Where Q is low, does not matter what P is.

{% include figure.html url="/_ml/prob_modelling/prob_basics/kl_asymmetry.png" description="Figure 2: Possible outcomes trying to match a multimodal Gaussian with a single Gaussian. (a) shows the result of a forward-KL optimization. (b) and (c) possible reverse KL results (depends on initialization). (Image from Bishop)"%}


## Hypothesis Testing

#### p-value
- Probability of observing a result at least as extreme as the ones obtained assuming $$\mathcal{H}_0$$ is correct.
- Gives a metric of how likely it is to have observed something "by chance".
- If $$p_{val}$$ is very small $$\rightarrow$$ it is very unlikely to have observed what we observed under $$\mathcal{H}_0$$ $$\rightarrow$$ must be rejected (it is probably wrong as our observation is very strange).

#### Statistical testing pipeline
Statistical testing is used to know whether an hypothesis is significant.
The main steps are as follows:

1. Define a null Hypothesis $$H_0$$ which will be rejected or not.
2. Define a **confidence** $$c$$ or **critical value** $$\alpha = 1 - c$$.
   - If $$p_{val} < \alpha \rightarrow \text{Reject hypothesis}$$
   - If $$p_{val} > \alpha \rightarrow \text{Do NOT reject hypothesis}$$

3. Determine the distribution our data follows.
4. Compute a suited test-statistic and p-value.
5. Reject/Don't reject hypothesis

**Common tests:** z-test, t-test, ANOVA, $$\chi^2$$-test...


{% include end-row.html %}


<!-- ## Variational inference

**NB**: _VI can be done using any distance, but reverse KL-divergence is useful for its tractability_

Consider an observation $$x$$ (will be the data) and a latent variable $$z$$ (will be the parameters of our model $$\theta$$).
Consider a parametrized approximation of $$P(z \mid x)$$ with params $$\omega$$: $$Q_\omega (z)$$.
This is:  $$Q_\omega (z)$$ is an approximation of our model parameters distribution.
Applying the Bayes rule on reverse KL divergence we get that:

\begin{equation}
D_{KL} (Q_\omega || P) = -
\sum Q_\omega (x) \log \frac{P (x, z)}{Q_\omega (x)} + \log P(x)
\end{equation}

Re-arranging:

\begin{equation}
\log P(x) = \sum Q_\omega (x) \log \frac{P (x, z)}{Q_\omega (x)} + D_{KL} (Q_\omega || P) 
\end{equation}

Where $$\log P(x)$$ is constant and $$D_{KL} (Q_\omega \mid \mid P)$$ is bounded by 0.
So minimizing $$D_{KL} (Q_\omega \mid \mid P)$$ is equivalent to maximizing $$\sum Q_\omega (x) \log \frac{P (x, z)}{Q_\omega (x)}$$.

$$\sum Q_\omega (x) \log \frac{P (x, z)}{Q_\omega (x)}$$ is known as **variational lower bound** or **evidence lower bound** (ELBO).

If we develop it by splitting the logarithm we obtain:

\begin{equation}
\max_\omega \sum Q_\omega \log P(y \mid X, \theta) - D_{KL} (Q_\omega (\theta) \mid \mid P (\theta))
\end{equation}

Where $$P (\theta)$$ is the prior of the model parameters distribution. Assuming the data samples are i.i.d, it becomes:

\begin{equation}
\max_\omega \sum Q_\omega \sum_i \log P(y_i \mid x_i, \theta) - D_{KL} (Q_\omega (\theta) \mid \mid P (\theta))
\end{equation}

This formulation can be used to optimize the parameters of an ANN from a Bayesian approach.
It is interesting to compare the objective function of a deterministic approach:

\begin{equation}
\min_\theta \sum_i l \left(f_\theta (x_i), y_i \right) + \Omega(\theta)
\end{equation} -->

