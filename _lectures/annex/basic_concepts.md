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

- $s_t$: **State** at time t.
- $o_t$: **Observation** at time t.
- $a_t$: **Action** at time t.
- $r_t$: **Reward** associated for taking action $a_t$ at state $s_t$.
- $\pi_{\theta} (a_t \mid o_t)$: Probability of taking action $a$  given the observation $o$ at time t according to the set of parameters $\theta$. The **Policy** essentially maps observations to actions.
    - Can either be stochastic or deterministic (1 action with prob 1, all others with prob 0).
    - Can either be from a partially observable or fully observable regime.

{% include figure.html url="/_lectures/annex/basics.png" description="Example of decision process elements" %}

**Observation vs State**: Observation is what the agent perceives. State are the underlying circumstances which give that observation, depends on the level of abstraction desired. States fully describe the world, observations only partly (not bijective) but a lot of times they are used indistinctly. 

## Markov Decision Processes (MDP)
**MDP:** It is a discrete-time stochastic control process which certifies the **Markov property**: the conditional probability distribution of future states of the process (conditional on both past and present states) depends only upon the present state, not on the sequence of events that preceded it.

{% include figure.html url="/_lectures/annex/markov.png" description="Graphical model of a Markov decision process" %}

**OBS:** If the environment transitions are known, the optimal policy can be found using **Dynamic Programming (DP)** approaches.

**OBS:** Observations do NOT necessarily satisfy Markov property but states do.