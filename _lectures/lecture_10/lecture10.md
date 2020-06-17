---
layout: lecture
title: "Lecture 10: Model-based Planning"
permalink: /lectures/lecture10
lecture-author: Sergey Levine
lecture-date: 2019
slides-link: http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-10.pdf
video-link: https://www.youtube.com/watch?v=pE0GUFs-EHI&list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A&index=11&
---

# Introduction

Until now, we assumed we did not know the dynamics of the environment.
Nevertheless often are known or can be learned.
How should we approach the problem in those cases?

## Deterministic environments:

Given a first state $$s_1$$, you can build a plan $$\{a_1, ..., a_T\}$$ defined by:

\begin{equation}
a_1, ..., a_T = argmax_{a_1, ..., a_T} \sum_t r(s_t, a_t)
\space \space s.t.
a_{t+1} = \mathcal{T} (s_t, a_t)
\end{equation}

To then simply blindly execute the actions.

## Stochastic environment open-loop:

What happens if we apply this same approach in a stochastic environment?
We can get a probability of a trajectory, given the sequence of actions:

\begin{equation}
p_\theta(s_1, ..., s_T \mid a_1, ..., a_T) =
p(s_1) \prod_t p(s_{t+1} \mid s_t, a_t)
\end{equation}

In this case, we want to maximize the expected reward:

\begin{equation}
a_1, ..., a_T = argmax_{a_1, ..., a_T}
E \left[ \sum_t r(s_t, a_t) \mid a_1, ..., a_T \right]
\end{equation}

**Problem:** Planning beforehand if the environment is stochastic may result in total different trajectories as the expected one.

## Stochastic environment closed-loop:

You define a policy $$\pi$$ instead of a plan:

\begin{equation}
\pi = argmax_{\pi} E_{\tau \sim p(\tau)} \left[ \sum_t r(s_t, a_t) \right]
\end{equation}

# Open-Loop planning

We can frame the planning as a simple optimization problem. If we write the plan $${a_1, ..., a_T}$$ as a matrix $$A$$, and the reward function as $$J$$, we just need to solve:

\begin{equation}
A = argmax_A J(A)
\end{equation}


## Stochastic optimization methods
Black-box optimization techniques.

### Guess & Check (Random Search)
1. Sample $$A_1,..., A_N$$ from some distribution $$p(A)$$ (e.g. uniform)
2. Choose $$A_i$$ based on $$argmax_i J(A_i)$$

### Cross-Entropy Method (CEM)
1. Sample $$A_1,..., A_N$$ from some distribution $$p(A)$$
2. Choose $$M$$ highest rewarding samples results $$A^1,...,A^M$$, *the elites*
3. Re-fit $$p(A)$$ to *the elites* $$A^1,...,A^M$$.

**OBS:** Not a great optimizer but easy to code and easy to parallelize. Works well for low-dimension systems (dimensionality < 64) and few time-steps.

**Improvements:** CMA-ES, implements momentum into CEM.

## Monte Carlo tree search (MCTS)
Handles better the stochastic case.




## Trajectory optimization
