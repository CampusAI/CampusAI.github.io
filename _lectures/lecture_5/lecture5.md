---
layout: lecture
title: "Lecture 5: Policy Gradients"
permalink: /lectures/lecture5
lecture-author: Sergey Levine
lecture-date: 2019
slides-link: http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf
video-link: https://www.youtube.com/watch?v=Ds1trXd6pos&list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A
---

In this lecture we focus on the case where we have an explicit, differentiable policy that maps
a given state to the correspondent distribution over actions. In deep RL this is tipically
achieved by a Neural Network, but other function approximation techniques may be used.

{% include figure.html url="/_lectures/lecture_5/NN_policy.png" description="Example of a policy modeled by a Convolutional Neural NetworkExample of a policy modeled by a Convolutional Neural Network" %}

#### Notation
Such a policy is parametrized by a set of parameters $\theta$ that can be, for example, the
weights of a Neural Network. Thus, $$\pi_{\theta}(a | s)$$ represents the probability of
chosing action $a$ given the current state $s$. We use $\pi_{\theta}(\tau)$ to represent
the probability of a trajectory $\tau$. We represent the total reward of a trajectory $\tau$
as $r(\tau)$


## Direct Policy Differentiation
We aim to maximize the objective function $J(\theta)$ of Eq. \ref{eq:objective}. We can do this
by exploiting the common gradient step technique. 
\begin{equation}
\label{eq:objective}
J(\theta) = E_{\tau \sim \pi_{\theta}(\tau)}[r(\tau)]
\end{equation}

Exploiting the **log gradient trick** we obtain that
\begin{equation}
\label{eq:objective_gradient}
\nabla_{\theta} J(\theta) = E_{\tau \sim \pi_{\theta}(\tau)}[
\nabla_{\theta}\log \pi_{\theta}(\tau) r(\tau)]
\end{equation}

Then, by plugging in $$\pi_{\theta}(\tau) = p(s_1) \prod_{t=1}^T \pi_{\theta}(a_t \vert s_t) p(s_{t+1} \vert s_t, a_t)$$ into the log in Eq.\ref{eq:objective_gradient} and observing that
the gradient is zero for all the terms not depending on $\theta$, we obtain

\begin{equation}
\label{eq:policy_gradient}
\nabla_{\theta}(\theta) = E_{\tau \sim \pi_{\theta}(\tau)}
\left [\left (\sum_{t=1}^T \nabla_{\theta}\log \pi_{\theta}(a_t \vert s_t) \right ) r(\tau) \right]
\end{equation}

A more detailed explanation of the steps that bring to this result can be found in the
[ANNEX TODO]()
