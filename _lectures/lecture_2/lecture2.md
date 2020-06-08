---
layout: lecture
title: "Lecture 2: Imitation Learning"
permalink: /lectures/lecture2
lecture-author: Sergey Levine
lecture-date: 2019
slides-link: http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-2.pdf
video-link: https://www.youtube.com/watch?v=TUBBIgtQL_k&list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A&index=3&t=3206s
---

# Sequential decision problems definitions

## Basics

- $s_t$: **State** at time t.
- $o_t$: **Observation** at time t.
- $a_t$: **Action** at time t.
- $r_t$: **Reward** associated for taking action $a_t$ at state $s_t$.
- $\pi_{\theta} (a_t \mid o_t)$: Probability of taking action $a$  given the observation $o$ at time t according to the set of parameters $\theta$. The **Policy** essentially maps observations to actions.
    - Can either be stochastic or deterministic (1 action with prob 1, all others with prob 0).
    - Can either be from a partially observable or fully observable regime.

{% include figure.html url="/_lectures/lecture_2/basics.png" description="Example of decision process elements" %}

**Observation vs State**: Observation is what the agent perceives. State are the underlying circumstances which give that observation, depends on the level of abstraction desired. States fully describe the world, observations only partly (not bijective) but a lot of times they are used indistinctly. 

## Markov Decision Processes (MDP)
**MDP:** It is a discrete-time stochastic control process which certifies the **Markov property**: the conditional probability distribution of future states of the process (conditional on both past and present states) depends only upon the present state, not on the sequence of events that preceded it.

{% include figure.html url="/_lectures/lecture_2/markov.png" description="Graphical model of a Markov decision process" %}

**OBS:** If the environment transitions are known, the optimal policy can be found using **Dynamic Programming (DP)** approaches.

**OBS:** Observations do NOT necessarily satisfy Markov property but states do.

# Imitation learning approach

## Behavioral cloning (BC)

**IDEA:** Record a lot of "expert" demonstrations and apply classic supervised learning to obtain a model to map observations to actions (policy).

{% include figure.html url="/_lectures/lecture_2/bc.png" description="Self-driving vehicle behavioral cloning workflow example." %}

<!-- **OBS:** Behavioral cloning is just a fancy way to say "supervised learning". -->

## Why doesn't this work?

### 1. Distributional shift

**Wrong actions change the data distribution:** A small mistake at the beginning makes the observations distribution to be different from the training data. This makes the policy to be more prone to error: it has not been trained on this  new distribution. This snowball effect keeps rising the error between trajectories over time.

{% include figure.html url="/_lectures/lecture_2/bc_problem.png" description="Representation of the distributional shift problem." %}

#### Improvements:
- Using some application-specific "hacks": [link](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)
- Adding noise to training trajectory so its more robust against errors.
- Adding a penalty for deviating (inverse RL idea)
- **DAgger** algorithm (**D**ataset **Aggre**gation)

##### DAgger algorithm

**Idea:** Collect training data from policy distribution instead of human distribution, using the following algorithm:

1. Train $\pi_{\theta} (a_t \mid o_t)$ on expert data: $\mathcal{D} = (o_1, a_1, ..., o_N, a_N)$
2. Run $\pi_{\theta} (a_t \mid o_t)$ to get a dataset $\mathcal{D}_\pi = (o_1, ..., o_N)$.
3. Ask expert to label $\mathcal{D}_\pi$ with actions $a_t$.
4. Aggregate $\mathcal{D} \leftarrow \mathcal{D} \cup \mathcal{D}_\pi$
5. Repeat

**Problem:** While it addresses the distributional shift problem, it is and unnatural way for humans to provide labels (we expect temporal coherence) $\Rightarrow$ Bad labels.

### 2. Non-Markovian behavior
If we see the same thing twice we won't act exactly the same. What happened in previous time-steps affects our current actions. Most decision we take are in a non-Markovian setup.

#### Improvements:
- We could feed the whole history to the model but the input would be too large to train robustly.
- We can use a RNN approach to account for the time dependency.

#### Problems:
- [Causal Confusion](https://arxiv.org/abs/1905.11979): Training models with history may exacerbate wrong causal relationships.

### 3. Multimodal behavior

In a continuous action space, if the parametric distribution chosen for our policy is not multimodal (e.g. a single Gaussian) the Maximum Likelihood Estimation (MLE) of the actions may be a problem:

{% include figure.html url="/_lectures/lecture_2/tree.png" description="While both -go left- and -go right- actions are ok, the averages action is bad." %}


#### Improvements:
- **Output a mixture of Gaussians**: Encode the action space as:
    - $\pi (a \mid o) = \sum_i w_i \mathcal{N} (\mu_i, \Sigma_i)$
- **Latent variable models**: Can be as expressive as we want: We can feed to the network a prior variable sampled from a known distribution. In this case, the policy training is harder but can be done using a technique like:
    - Conditional variational autoencoder
    - Normalizing flow/realNVP
    - Stein variational gradient descent
- **Autoregressive discretization**: Convert the continuous action space into a discrete one using neural nets. The idea is to sequentially discretize one dimension at a time:
    1. Feed-forward policy network to obtain each action continuous distribution.
    2. Split first dimension into bins and sample as a categorical distribution.
    3. Feed-forward this sampled value into a new small NN with inputs the $n-1$ other actions and outputs the $n-1$ other actions again.
    4. Repeat from 2. until all actions are discretized.

# Theory

[Drone](https://idsia-robotics.github.io/files/publications/RAL16_Giusti.pdf) example.