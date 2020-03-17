---
layout: paper
title: Neural Network Surgery with Sets
category: other
permalink: /papers/NN_surgery_sets
paper-author: Jonathan Raiman, Susan Zhang, Christy Dennison
post-author: Oleguer Canal
paper-year: 2019
paper-link: https://arxiv.org/abs/1912.06719
---

- Performing exploration on **input features** and **architecutres** for Neural Networks is a highly time-consuming task (often prohibiting).
After changing input features or architecture, the network parameters have to be fully re-trained from scratch.
- This paper introduces a solution which enables the **transfer** of learned parameters from a model to a modification of it.
- This allows to **continously train** the model while performing architecture modifications.

## Idea
Given a model $F_{old}$ with parameters $\Theta_{old}$, and another model, result of a modification $F_{new}$ with parameters $\Theta_{new}$ we wish to find a **mapping**:

\begin{equation}
M : \Theta_{old} \rightarrow \Theta_{new}
\end{equation}

This can be achieved in 2 steps:

1. Map input features ($x_{model}^{in}$) to all model parameters that relate ($[\theta_{0}, \theta_{1}, ...]$), for both models.
These functions are refered as $\phi_{old}$ and $\phi_{new}$, and take the shape of a **lookup table** with:
- **keys**: each $x_{model}^{in}(i)$
- **values**: list of parameters $[\theta_{i_0}, \theta_{i_1}, ...]$ which are dependent on input feature $x^{in}(i)$.

2. Compare $\phi_{old}$ and $\phi_{new}$ and use it to compute how to initialize parameters in the new model:
- Newly introduced features will not share keys in lookup tables $\phi_{old}$ and $\phi_{new}$. We can use this mismatch to detect which parameters to re-initialize.
- Additionally, we can detect keys shifts to know which old parameters should be copied to the new model.


<!-- {% include figure.html url="/assets/images/[PATH]" description="[DESCRIPTION]" %} -->

#### Parameter Mapping
The authors present 2 **equivalent** approaches to compute the lookup tables of mentioned $\phi_{old}$ mappings:
- **Gradient Mapping**: Based on output differentiation w.r.t. each parameter for an input vector of all zeors but a single one (key position in the lookup table).
- **Boolean Logic Mapping**: Based on explicitly tracking when a particular input feature is involved in an operation with a model parameter. Every model parameter mantains a list of boolean flags where a true value means involvement with a given input feature.

Results show that Boolean Logic Mapping outperforms both in computation time and robustness Gradient Mapping.

## Contribution
 - **Parameter Mapping Algorithm** able to detect 100% of interactions within ~37 seconds (on a 2.9 GHz Intel I7 CPU)
 - **Used while training OpenAi Five Dota2 agent**: This approach helped adapt OpenAi Five agent's model. They introduced around 20 major architecture changes over the 10-months training period.

## Weaknesses
 - **Need to scramble through all weights** to detect input-parameter relationships. Is there a better way to robustly continue training a model without doing so?

## Additional Information
- Two Minute Papers reviews this approach on a great [video](https://www.youtube.com/watch?v=62Q1NL4k8cI&t=0s)