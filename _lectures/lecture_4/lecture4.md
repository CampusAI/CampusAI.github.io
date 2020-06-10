---
layout: lecture
title: "Lecture 4: Introduction to Reinforcement Learning"
permalink: /lectures/lecture4
lecture-author: Sergey Levine
lecture-date: 2019
slides-link: http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-4.pdf
video-link: https://www.youtube.com/watch?v=w_IIP-swuVo&list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A
---

## The Reinforcement Learning Framework
Reinforcement Learning deals with **agents** interacting in a certain **environment**, observing
its **state**, performing **actions** and obtaining **rewards**, with the goal of learning an
optimal **policy**, i.e. a mapping from observations to actions, that maximizes the total reward.

### Markov Decision Process
The interaction of the agent with the environment is defined as a Markov Decision Process.
The elements of this MDP are:

- **State space $S$**: the set of possible states $s_i \in S$ of the environment. It is
    important to keep in mind that often what the agent sees is not the actual state of the
    environment, but rather a -possibly noisy- observation of it, which is often written
    $o \in O$.
- **Action space $A$**: the set of possible actions $a_i \in A$.
- **Reward $r(s, a)$**: reward function that maps state-action pairs to the correspondent scalar
    rewards. Rewards represent how good or bad are the given state-action pairs.
- **Transition operator $\mathcal{T}$**: the operator that encodes the transition probabilities
    given state and action, i.e. $$\mathcal{T}_{s', s, a} = p(s_{t+1} = s' | s_t = s, a_t = a)$$.
- **Policy $$\pi(a, s)$$**: a distribution over the actions taken by the agent.
    $$\pi(a, s)$$ represents the probability of choosing action $a$ given that the
    current state is $s$. This policy is usually dependent on some set of parameters $\theta$ (e.g. weights in a ANN). We therefore refer to the policy as: $\pi_{\theta}(a, s)$.

Note that, for a given policy
$\pi_{\theta}$, the probability of a **trajectory** $$\tau = (s_1, a_1, s_2, a_2, \:...)$$
is given by the induced Markov chain on the joint space $S$ x $A$

\begin{equation}
p_{\theta}(\tau) = p(s_1, a_1, r_1, s_2, a_2, \:...) =
p(s_1)\prod_{t=1}^{T}\pi_{\theta}(a_t, s_t)p(s_{t+1} | s_t, a_t)
\end{equation}

It is important to note that, given $$\mu^{(t)} = (p(s_1 ,a_1), p(s_1, a_2),  ... p(s_n, a_m))$$,
the joint distribution of states and actions at
time-step $t$, we obtain the distribution $\mu^{(t+1)}$ by applying the transition operator
\begin{equation}
\mu^{(t+1)} = \mathcal{T}(\mu^{(t)})
\end{equation}

In the same way we obtain that, for an horizon $k$, $$\mu^{(t+k)} = \mathcal{T}^k (\mu^{(t)})$$.
If such a Markov chain is **ergodic** -i.e. any state can be reached by any other state in a finite
number of steps with a non-zero probability- then, for an infinite horizon, we obtain a **stationary
distribution** $$p_{\theta}(s, a)$$ in the limit. 

In a stationary distribution $\mu$, applying the transition operator does not change the distribution: 
$$\mu = \mathcal{T}\mu$$.

### The goal of Reinforcement Learning
Now that we defined a distribution over the states and actions, we need to define an
**objective function** that we aim to maximize.

#### Finite Horizon
In finite horizon problems the agent lives for $T$ time-steps. We aim to achieve the maximum 
cumulative reward in expectation over the distribution $p(\tau)$:
\begin{equation}
\label{eq:finite_hor}
\theta^* = \arg \max_{\theta} E_{\tau \sim p_{\theta}(\tau)}
\left [ \sum_{t=1}^T r(s_t, a_t) \right ]
= \arg \max_{\theta} E_{(s_t, a_t) \sim p_{\theta}(s_t, a_t)}\left[ r(s_t, a_t) \right]
\end{equation}
where $p_{\theta}(s_t, a_t)$ is the **state-action marginal** distribution that we obtain by applying
the $\mathcal{T}$ operator $t$ times.


#### Infinite Horizon
We now deal for the case where the horizon $T$ of Eq. \ref{eq:finite_hor} is infinite. As we said, for
$$t\rightarrow \infty$$, the distribution of states and actions $$p_{\theta}(s_t, a_t)$$ converges to
a stationary distribution $$p_{\theta}(s, a)$$. We then re-write Eq.\ref{eq:finite_hor} as
\begin{equation}
\label{eq:inf_hor}
\theta^* = \arg \max_{\theta} \frac{1}{T} \sum_{t=1}^{\infty}
E_{(s_t, a_t) \sim p_{\theta}(s_t, a_t)} [r(s_t, a_t)] \rightarrow
E_{(s, a) \sim p_{\theta}(s, a)}[r(s, a)]
\end{equation}

where we divide by 1/T to make the sum finite. This is called the **undiscounted average return** formulation of RL.


#### The Q and V functions
We define the **Q function** as
\begin{equation}
Q^{\pi}(s_t, a_t) = \sum_{t'=t}^T E_{\pi_{\theta}}[ r(s_{t'}, a_{t'}) | s_t, a_t]
\end{equation}
i.e. $$Q^{\pi}(s_t, a_t)$$ is the expected total reward obtained by starting in state $s_t$ and doing
action $a_t$ while following policy $\pi$.

Similarly, we define the **Value Function** as
\begin{equation}
V^{\pi} (s_t) = \sum_{t'=t}^T E_{\pi_{\theta}}[r(s_{t'}, a_{t'}) | s_t] = 
E_{a_t \sim \pi_{\theta}(a_t| s_t)}[Q^{\pi}(s_t, a_t)]
\end{equation}
i.e. $$V^{\pi}(s_t)$$ is the total reward expected by starting in $$s_t$$ and following the policy
$\pi$. Note that $$V^{\pi}(s_1)$$ is our RL objective.


### RL algorithms
{% include figure.html url="/_lectures/lecture_4/RL_algorithm.png" description="Figure 1: The anatomy of a RL algorithm" %}


Given the objective of Eq. \ref{eq:finite_hor} or \ref{eq:inf_hor}, or some other variants such as
**discounted total reward**, we can exploit different algorithms to optimize a policy:
- **Policy Gradients**: directly differentiate the objective function
- **Value Based**: estimate the value function or the $Q$ function of the optimal policy
- **Actor-Critic**: estimate the value or $Q$ function of the current policy, and use it to improve
the policy
- **Model Based**: estimate the transition model of the environment and use it for planning,
improving the policy, and more.


An important distinction to make is the one between **on-policy** and **off-policy** algorithms:
- **Off-Policy**: able to improve the policy using samples collected from different policies.
    These algorithms are more **sample efficient** because they need less samples from the environment.
- **On-Policy**: need to generate new samples every time the policy is changed, therefore require more
    samples.

It is important to note that better sample efficiency does not imply a better model. In some cases,
generating new samples may be way faster than updating the policy (for example, when using big neural
networks). Different algorithms make different assumptions on the state and action spaces as well as
the nature of the task (episodic vs continuous). Moreover, **convergence is not always guaranteed** in
many cases! One should always chose the algorithm by taking all this and much more factors into account.
