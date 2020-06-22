---
layout: lecture
title: "Lecture 7: Value Function Methods"
permalink: /lectures/lecture7
lecture-author: Sergey Levine
lecture-date: 2019
slides-link: http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-7.pdf
video-link: https://www.youtube.com/watch?v=doR5bMe-Wic&list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A
---

In the previous lecture ([Actor-Critic Algorithms](/lectures/lecture6)) we learned how to
improve the policy by taking gradient steps proportional to an Advantage Function
$$A^{\pi}(s_t, a_t)$$ that tells us *how much better* is action $$a_t$$ than the average action
in state $$s_t$$ according to the policy $$\pi$$. We defined the Advantage Function as
\begin{equation}
\label{eq:advantage}
A^{\pi}(s_t, a_t) = r(s_t, a_t) + E_{s_{t+1} \sim p(s_{t+1} \vert s_t, a_t)} \left [
\gamma V^{\pi}(s_{t+1}) \right] - V^{\pi}(s_t)
\end{equation}

### What if we could omit the Policy Gradient?
If we have a good estimation of the Advantage Function, we do not need an explicit policy: we
can just choose actions accordingly to
\begin{equation}
a_t = \arg\max_{a}A^{\pi}(a, s_t)
\end{equation}
i.e. we take the action that would lead to the highest reward by taking that action and then
following $$\pi$$. 

### Policy Iteration
The considerations above lead us to the **Policy Iteration** idea:

Repeat:
1. Evaluate $$A^{\pi}(s_t, a_t)$$
2. Set $$\pi(s_t) = \arg\max_{a}A^{\pi}(s_t, a)$$

The policy $$\pi$$ is now **deterministic** as it is an $$\arg\max$$ policy and, knowing
$$A^{\pi}$$ we can improve it in a straightforward way. We now need to understand
how to evaluate $$A^{\pi}$$.

## Evaluating $$V^{\pi}$$
Following Eq. \ref{eq:advantage} we now evaluate $$A^{\pi}$$ by evaluating $$V^{\pi}$$.
We make some strong assumptions that we will relax later, but help us defining the problem:
- **We know** the environment dynamics $$p(s_{t + 1} \vert s_t, a_t)$$ and rewards
  $$r(s_t, a_t)$$.
- The action space $A$ and state space $S$ are **discrete** and **small enough** to be
  stored in a tabular form.

We can store the whole $$V^{\pi}(\pmb{s})$$ and perform a **bootstrapped update**
\begin{equation}
V^{\pi}(\pmb{s}) \leftarrow r(\pmb{s}, \pi(\pmb{s})) + \gamma
E_{s_{t+1} \sim p(s_{t+1} \vert s_t, a_t)} \left[ V^{\pi}(\pmb{s}_{t+1}) \right]
\end{equation}
where we denote all possible states with bold $$\pmb{s}$$. Note that we are using the current
estimate of $$V^{\pi}$$ when computing the expectation for the next states $$\pmb{s}_{t+1}$$.
Since we are assuming to know the transition probabilities, the expected value can be computed
analytically.

## Evaluating $$Q^{\pi}$$
By analyizing the deterministic policy

\begin{equation}
\pi(\pmb{a}_t \vert \pmb{s}_t) = \arg\max_a A^{\pi}(\pmb{s}_t, \pmb{a}_t)
\end{equation}
