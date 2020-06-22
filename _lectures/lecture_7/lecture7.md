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

We make some strong assumptions that we will relax later, but help us defining the problem:
- **We know** the environment dynamics $$p(s_{t + 1} \vert s_t, a_t)$$ and rewards
  $$r(s_t, a_t)$$.
- The action space $A$ and state space $S$ are **discrete** and **small enough** to be
  stored in a tabular form.

During this lecture we will using the bold $$\pmb{s}$$ notation to refer to all possible states
$$s \in S$$, and an assignment $$V(\pmb{s}) \leftarrow something$$ means that the assignment is
performed *for each $$s \in S$$*. 

## Evaluating $$V^{\pi}$$
Following Eq. \ref{eq:advantage} we decide to evaluate $$A^{\pi}$$ by evaluating $$V^{\pi}$$.

We can store the whole $$V^{\pi}(\pmb{s})$$ and perform a **bootstrapped update**
\begin{equation}
V^{\pi}(\pmb{s}) \leftarrow r(\pmb{s}, \pi(\pmb{s})) + \gamma
E_{s_{t+1} \sim p(s_{t+1} \vert s_t, a_t)} \left[ V^{\pi}(\pmb{s}_{t+1}) \right]
\end{equation}
Note that we are using the current estimate of $$V^{\pi}$$ when computing the expectation
for the next states $$\pmb{s}_{t+1}$$. Since we are assuming to know the transition
probabilities, the expected value can be computed analytically. This leads us to the
**Policy Iteration** algorithm.

#### Policy Iteration

Repeat:
1. Evaluate $$V^{\pi}(\pmb{s})$$
2. Set $$\pi(\pmb{s}) = \arg\max_aA^{\pi}(s_t, a)$$ (from Eq. \ref{eq:advantage})

The policy $$\pi$$ is now **deterministic** as it is an $$\arg\max$$ policy. The policy $$\pi$$
is ensured to improve every time we perform the update in step 2 (if it is not already optimal)
thanks to the **Policy Improvement Theorem**
([Sutton & Barto, Sec 4.2](http://incompleteideas.net/book/bookdraft2017nov5.pdf)).

## Evaluating $$Q^{\pi}$$
By analyizing the deterministic argmax policy where $$A^{\pi}$$ is given by Eq.
\ref{eq:advantage}, we observe that the subtracted baseline $$V^{\pi}(s_t)$$ is independent
from the chosen action, and the $$\arg\max$$ step is therefore equivalent to

\begin{equation}
\pi(\pmb{a}_t \vert \pmb{s}_t) = \arg\max_a A^{\pi}(\pmb{s}_t, \pmb{a}_t) = 
\arg\max_aQ^{\pi}(\pmb{s}_t, \pmb{a}_t)
\end{equation}

since $$r(\pmb{s}_t, \pmb{a}_t) + \gamma E[V^{\pi}(\pmb{s}_{t+1})] =
Q^{\pi}(\pmb{s}_t, \pmb{a}_t)$$. If we stick to our assumption that we know the environment
dynamics, we store the $$Q^{\pi}$$ values in a table and we can create a new algorithm called
**Value Iteration**.

#### Value Iteration
The **Value Iteration** algorithm is now straightfoward:

Repeat:
1. Set $$Q(\pmb{s}, \pmb{a}) \leftarrow r(\pmb{s},\pmb{a}) + \gamma E[V^(\pmb{s}')]$$
2. Set $$V(\pmb{s}) \leftarrow \max_{\pmb{a}} Q(\pmb{s}, \pmb{a})$$

Until policy $$\pi = \arg\max_a Q(s, a)$$ does not change anymore.

As long as we are fully storing the $$Q$$ values, the algorithm is ensured to converge to the
optimal value function.

## Approximating value with Neural Networks
Storing a value for each posible state is often not possible because the state space is too big
or not discrete. We can then approximate the value of a state using a **Neural Network** with
parameters $$\phi$$. We define the **loss** as
\begin{equation}
\mathcal{L}(\phi) = \frac{1}{2} \vert\vert V_{\phi}(\pmb{s}) - \max_a Q^{\pi}(\pmb{s}, \pmb{a})
\vert\vert
\end{equation}
We can then train a network to approximate the value function and obtain the **Fitted Value
Iteration** algorithm.

#### Fitted Value Iteration
We are still assuming that we know the transition dynamics of the environment and that the
state space is finite.

Repeat:
1. Set $$\pmb{y}_i \leftarrow max_{a_i} \left( r(\pmb{s}_i, \pmb{a}_i) + \gamma
   E\left[ V_{\phi}(\pmb{s}_{i}')\right] \right)$$
2. Set $$\phi \leftarrow \arg\min_{\phi} \frac{1}{2} \sum_{i} \vert\vert
   V_{\phi}(\pmb{s}_i) - \pmb{y}_i \vert \vert ^2$$

Note that we still need to iterate trough all the possible states, although now we do not need
to store a value for each, and we need to know the transition dynamics to compute
$$E[V_{\phi}(\pmb{s}_i')]$$.

### Unknown transition dynamics and infinite states
The algorithm above has an issue that prevents us from relaxing the assumptions we made: we need
to perform a $$\max$$ operation on $$E[V_{\phi}]$$ over all possible actions. If we are
learning $$V_{\phi}$$ in step 2, we cannot know which action leads to the highest value
without knowing the transition dynamics. For this reason, we will **learn the $$Q$$ function
instead**. By approximating the Q values with $$Q_{\phi}$$ we can now take the $$\max$$ of our
approximation and use it to estimate $$E[V(\pmb{s}')] \approx \max_{a'} Q(\pmb{s}', \pmb{a}')$$.
By relaxing the assumption of knowing the transitions, we now must rely on **sampling** in order
to compute the expectation. We relax the assumption of having a finite number of states by
exploiting the sampled states instead of iterating trough all the possible states.

Therefore, we can define the **Fitted Q Iteration** algorithm.

#### Fitted Q Iteration
The following is the Fitted Q Iteration algorithm along with the parameters of each step.
{% include figure.html url="/_lectures/lecture_7/fitted_q_iteration.png" %}

There are a few observations that we can make to better understand what this algorithm does.
- **Off-Policy**: The Fitted Q Iteration algorithm is off-policy. In fact, when fitting the
  $$Q_{\phi}$$ estimator, the targets $$\pmb{y}_i$$ are built by taking the $$\max$$ of
  $$Q_{\phi}$$ over all possible actions. Therefore, the stored *transitions* act as a dataset
  From which we learn the $$Q$$ function. It is important to note that while the learning from
  the obtained transitions is off-policy, the collection of those is **on-policy**. After
  fitting the $$Q_{\phi}$$ we need to collect new transitions to populate our dataset with
  states and rewards that our improved policy is now able to reach.
- **Does it converge?** The tabular policy iteration was ensured to converge, but this is
  not true anymore when we leave the tabular representation and use a Neural Network.

We can derive an **online** version of the algorithm that we call **Online Q-Iteration** or
**Q-Learning**:
{% include figure.html url="/_lectures/lecture_7/online_q_learning.png" %}


### Exploration with Q Learning
In step 1 of the **Fitted Q Iteration** and **Q-Learning** we collect one or more transitions.
This step is really important as it is the step in which we collect the data that will be used
when fitting the $$Q_{\phi}$$ network. We cannot directly use the $$\arg\max$$ policy because it
lacks of **exploration**: it always choses the action for which the estimated value is higher,
preventing the exploration of all actions. We therefore need some kind of **stochastic
exploration policies** that allow us to explore while still taking, on average, good actions
that allow us to reach new important states. In a videogame, for example, we need to explore
multiple actions in order to learn, but we also need to take good actions in order to reach
new levels and thus observe new interesting states.

#### $$\epsilon$$-Greedy policies
In $$\epsilon$$-greedy policies we take a random action with probability $$\epsilon$$, and the
$$\arg\max$$ action with probability $$1-\epsilon$$:

$$
\pi(a_t \vert s_t) = 
\begin{cases}
1 - \epsilon, & \text{if} \ a_t = \arg\max_a Q_{\phi}(s_t, a) \\
\frac{\epsilon}{\vert A \vert -1} & \text{otherwise}
\end{cases}
$$

This **exploration** policy ensures that we explore all actions but still, on average, 
perform good actions that will make us obtain *good* samples as the policy advances towards
most interesting states. 

#### Boltzmann Exploration
In Boltzmann exploration we explore actions proportionally to a transformation of their Q value.

$$
\pi(a_t \vert s_t) \propto \exp Q_{\phi}(s_t, a_t)
$$

This allows us to take random actions in a way that is more oriented towards our current
$$Q_{\phi}$$ estimate
