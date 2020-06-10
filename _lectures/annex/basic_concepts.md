---
layout: lecture
title: "Annex 1: Notation and Basic Concepts"
permalink: /lectures/basic_concepts
lecture-author: Sergey Levine
lecture-date: 2019
slides-link: http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-1.pdf
video-link: https://www.youtube.com/watch?v=SinprXg2hUA&list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A&index=2&t=0s
---

# Sequential decision problems definitions
## Basics

- **State space** $$S$$: the set of possible states $s_i \in S$ of the environment. It is
    important to keep in mind that often what the agent sees is not the actual state of the
    environment, but rather a -possibly noisy- **observation** of it, which is often written
    $o \in O$.
- **Action space** $$A$$: the set of possible actions $a_i \in A$.
- **Reward $r(s, a)$**: reward function that maps state-action pairs to the correspondent scalar
    rewards. Rewards represent how good or bad are the given state-action pairs.
- **Transition operator $$\mathcal{T}$$**: the operator that encodes the transition probabilities
    given state and action, i.e. $$\mathcal{T}_{s', s, a} = p(s_{t+1} = s' | s_t = s, a_t = a)$$.
- **Policy $$\pi_{\theta}(a, s)$$**: a distribution over the actions taken by the agent.
    $$\pi_{\theta}(a, s)$$ represents the probability of chosing action $a$ given that the
    current state is $s$. The $\theta$ subscript means that the policy is parametrized by $\theta$.
    A policy can be **stochastic** or **deterministic**, and can operate on a fully or partially
    observable environment.

{% include figure.html url="/_lectures/annex/basics.png" description="Example of decision process elements" %}

**Observation vs State**: Observation is what the agent perceives. State are the underlying circumstances which give that observation, depends on the level of abstraction desired. States fully describe the world, observations only partly (not bijective) but a lot of times they are used indistinctly. 

## Markov Decision Processes (MDP)
**MDP:** It is a discrete-time stochastic control process which certifies the **Markov property**: the conditional probability distribution of future states of the process (conditional on both past and present states) depends only upon the present state, not on the sequence of events that preceded it.

{% include figure.html url="/_lectures/annex/markov.png" description="Graphical model of a Markov decision process" %}

**OBS:** If the environment transitions are known, the optimal policy can be found using **Dynamic Programming (DP)** approaches.

**OBS:** Observations do NOT necessarily satisfy Markov property but states do.
