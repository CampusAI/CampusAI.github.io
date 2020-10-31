---
layout: lecture
title: "Lecture 2: Imitation Learning"
permalink: /lectures/lecture2
lecture-author: Sergey Levine
lecture-date: 2019
post-author: Oleguer Canal
slides-link: http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-2.pdf
video-link: https://www.youtube.com/watch?v=TUBBIgtQL_k&list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A&index=3&t=3206s
---
<!--
Disclaimer and authorship:
This article is provided for free only for your personal informational and entertainment purposes. No commercial use of it is allowed.

Please note there might be mistakes. We would be grateful to receive (constructive) criticism if you spot any. You can reach us at: ai.campus.ai@gmail.com or directly open an issue on our github repo: https://github.com/CampusAI/CampusAI.github.io

If considering to use the text please cite the original author/s of the lecture/paper.
Furthermore, please acknowledge our work by adding a link to our website: https://campusai.github.io/ and citing our names: Oleguer Canal and Federico Taschin.
-->
{% include start-row.html %}

# Behavioral cloning (BC)

**IDEA:** Record a lot of "expert" demonstrations and apply classic supervised learning to obtain a model to map observations to actions (policy).

{% include end-row.html %}
{% include start-row.html %}

{% include figure.html url="/_rl/lecture_2/bc.png" description="Self-driving vehicle behavioral cloning workflow example." %}

{% include annotation.html %}

Behavioral cloning is a type of Imitation Learning -a general term to say "learn from expert demonstrations".

Behavioral cloning is just a fancy way to say "supervised learning".

{% include end-row.html %}
{% include start-row.html %}


# Why doesn't this work?

## 1. Distributional shift

**Wrong actions change the data distribution:** A small mistake at the beginning makes the observations distribution to be different from the training data. This makes the policy to be more prone to error: it has not been trained on this  new distribution. This snowball effect keeps rising the error between trajectories over time.

{% include figure.html url="/_rl/lecture_2/bc_problem.png" description="Representation of the distributional shift problem." %}

### Improvements:
- Using some application-specific "hacks": [Self-driving car](https://devblogs.nvidia.com/) and [Drone](https://idsia-robotics.github.io/files/publications/RAL16_Giusti.pdf) trained with BC.
- Adding noise to training trajectory so its more robust against errors.
- Adding a penalty for deviating (inverse RL idea)
- **DAgger** algorithm (**D**ataset **Aggre**gation)

#### DAgger algorithm

**Idea:** Collect training data from policy distribution instead of human distribution, using the following algorithm:

1. Train $$\pi_{\theta} (a_t \mid o_t)$$ on expert data: $$\mathcal{D} = (o_1, a_1, ..., o_N, a_N)$$
2. Run $$\pi_{\theta} (a_t \mid o_t)$$ to get a dataset $$\mathcal{D}_\pi = (o_1, ..., o_N)$$.
3. Ask expert to label $$\mathcal{D}_\pi$$ with actions $$a_t$$.
4. Aggregate $$\mathcal{D} \leftarrow \mathcal{D} \cup \mathcal{D}_\pi$$
5. Repeat

**Problem:** While it addresses the distributional shift problem, it is and unnatural way for humans to provide labels (we expect temporal coherence) $$\Rightarrow$$ Bad labels.

## 2. Non-Markovian behavior
If we see the same thing twice we won't act exactly the same. What happened in previous time-steps affects our current actions. Most decision we take are in a non-Markovian setup. This makes the training much harder.

### Improvements:
- We could feed the whole history to the model but the input would be too large to train robustly.
- We can use a RNN approach to account for the time dependency.

### Problems:
- [Causal Confusion](https://arxiv.org/abs/1905.11979): Training models with history may exacerbate wrong causal relationships.

## 3. Multimodal behavior

In a continuous action space, if the parametric distribution chosen for our policy is not multimodal (e.g. a single Gaussian) the Maximum Likelihood Estimation (MLE) of the actions may be a problem:

{% include figure.html url="/_rl/lecture_2/tree.png" description="While both 'go left' and 'go right' actions are ok, the average action is bad." %}


### Improvements:
- **Output a mixture of Gaussians**: $$\pi (a \mid o) = \sum_i w_i \mathcal{N} (\mu_i, \Sigma_i)$$
- **Latent variable models**: Can be as expressive as we want: We can feed to the network a prior variable sampled from a known distribution. In this case, the policy training is harder but can be done using a technique like:
    - Conditional variational autoencoder
    - Normalizing flow/realNVP
    - Stein variational gradient descent
- **Autoregressive discretization**: Convert the continuous action space into a discrete one using neural nets. The idea is to sequentially discretize one dimension at a time:
    1. Feed-forward policy network to obtain each action continuous distribution.
    2. Split first dimension into bins and sample as a categorical distribution.
    3. Feed-forward this sampled value into a new small NN with inputs the $$n-1$$ other actions and outputs the $$n-1$$ other actions again.
    4. Repeat from 2. until all actions are discretized.

# Quantitative analysis

Defining the cost function: $$c(s, a) = \delta_{a \neq \pi^*(s)}$$ (1 when the action is different from the expert).

And assuming that the probability of making a mistake on a state sampled from the training distribution is bounded by $$\epsilon$$: $$\space \space \pi_\theta (a \neq \pi^* (s) \mid s) \leq \epsilon \space\space \forall s \sim p_{train}(s)$$ 

### Case: $$p_{train}(s) \simeq p_{\theta}(s)$$

{% include end-row.html %}
{% include start-row.html %}

\begin{equation}
E \left[ \sum_t c(s_t, a_t) \right] = O(\epsilon T)
\end{equation}

{% include annotation.html %}

This would be the case if DAgger algorithm correctly applied, where the training data distribution converges to the trained policy one.

{% include end-row.html %}
{% include start-row.html %}


### Case: $$p_{train}(s) \neq p_{\theta}(s)$$

We have that: $$p_\theta (s_t) = (1-\epsilon)^t p_{train} (s_t) + (1 - (1 - \epsilon)^t) p_{mistake} (s_t)$$

Where $$p_{mistake} (s_t)$$ is a state probability distribution different from $$p_{train} (s_t)$$. In the worst case, the total variation divergence: $$\mid p_{mistake} (s_t) - p_{train} (s_t) \mid = 2$$

{% include end-row.html %}
{% include start-row.html %}

Therefore:

\begin{equation}
\sum_t E_{p_\theta (s_t)} [c_t] = O(\epsilon T^2)
\end{equation}

{% include annotation.html %}

The error expectation grows quadratically over time!! More details on: [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://arxiv.org/abs/1011.0686)

{% include end-row.html %}
