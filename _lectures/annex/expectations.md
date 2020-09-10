---
layout: article
title: "Annex 2: Policy Expectations, Explained"
permalink: /lectures/policy_expectations
lecture-author: None
lecture-date: 2020
post-author: Oleguer Canal
slides-link: /lectures/policy_expectations
video-link: /lectures/policy_expectations
---

Given each possible trajectory in the studied environment: $\tau = (s_1, a_1, s_2, a_2, \:...) \in \mathrm{T}$, we can compute its cumulative reward:

\begin{equation}
R(\tau) = \sum_{(s, a) \in \tau} r(s, a) \equiv \sum_{t} r(s_t, a_t)
\end{equation}

Similarly, given any policy $\pi_\theta$, we know that this trajectory has a certain probability of happening given by:

\begin{equation}
p_{\pi_\theta} (\tau) \equiv  p_{\theta} (\tau) = p(s_1)\prod_{t=1}^{T}\pi_{\theta}(a_t, s_t)p(s_{t+1} | s_t, a_t)
\end{equation}

The policy expected reward is nothing more than the sum for all possible trajectories probabilities times their reward:

\begin{equation}
E_{\tau \sim p_{\theta}(\tau)} \left[ R(\tau) \right] =
\sum_{\tau \in \mathrm{T}} p_{\pi_\theta} (\tau) \cdot R(\tau)
\end{equation}

This will be written in different ways throughout the lectures. For instance in an abuse of notation:

\begin{equation}
E_{\tau \sim p_{\theta}(\tau)} \left[ R(\tau) \right] \equiv
E_{\tau \sim p_{\theta}(\tau)}
\left [ \sum_{t=1}^T r(s_t, a_t) \right ]
\end{equation}

Notice that instead of considering the rewards and probabilities of entire trajectories, we could equivalently evaluate the probability of "being" in each (state, action) pair at a given time given some policy $\pi_\theta$.
In this case we could write the expected reward as:

\begin{equation}
\sum_{t} E_{(s_t, a_t) \sim p_{\theta}(s_t, a_t)}\left[ r(s_t, a_t) \right] =
\sum_{t} \sum_{(s, a) \in S\times A} p(s_t=s, a_t=a \mid \pi_\theta) \cdot r(s_t=s, a_t=a) 
\end{equation}
