---
layout: article
title: "Annex 5: Policy Gradients"
permalink: /lectures/policy_gradients_annex
post-author: Federico Taschin
---

### Obtaining the Policy Gradient
Here we explain the steps to obtain the gradient of the RL objective in a form that we are able
to compute. Note that this is not the proof of the Policy Gradient theorem, that can be found in
[Sutton & Barto, 2017, Section 13.1](http://incompleteideas.net/book/bookdraft2017nov5.pdf).
The objective function that we aim to maximize is
\begin{equation}
J(\theta) = E_{\tau \sim \pi_{\theta}(\tau)} [r(\tau)]
\end{equation}
where $$\pi_{\theta}(\tau)$$ is the probability distribution over the trajectories and
$$r(\tau)$$ is the total reward of a given trajectory $$\tau$$. Then, we are interested in the
gradient
\begin{equation}
\label{eq:target_grad}
\nabla_{\theta}J(\theta) = \nabla_{\theta} \int \pi_{\theta}(\tau) r(\tau) d\tau
= \int \nabla_{\theta} \pi_{\theta}(\tau) r(\tau) d\tau
\end{equation}

In which we expanded the expected value with the integral definition and we moved the gradient
operator inside the integral. We are allowed to perform this last operation thanks to the
[Leibniz Integral Rule](https://en.wikipedia.org/wiki/Leibniz_integral_rule) as long as both
$$\pi_{\theta}$$ and $$\nabla_{\theta}\pi_{\theta}$$ are continuous in $\theta$ and $\tau$. 
We now exploit the fact that
\begin{equation}
\label{eq:log_grad_trick}
\nabla_{\theta}\log\pi_{\theta}(\tau) = \frac{\nabla_{\theta}\pi_{\theta(\tau)}}
{\pi_{\theta}(\tau)}
\end{equation}

We substitute $$\nabla_{\theta}\pi_{\theta}(\tau) =
\pi_{\theta}(\tau) \nabla_{\theta}\log\pi_{\theta}(\tau)$$ into Eq. \ref{eq:target_grad} to
obtain
\begin{equation}
\label{eq:pg_grad_tau}
\nabla_{\theta}J(\theta) = \int \pi_{\theta}(\tau) \nabla_{\theta}\log\pi_{\theta}(\tau)
r(\tau) d\tau = E_{\tau \sim \pi_{\theta}(\tau)}\left[ 
\nabla_{\theta}\log\pi_{\theta}(\tau)r(\tau)
\right]
\end{equation}
in which we observed that the resulting integral is an expectation over the trajectories.
We recall that
\begin{equation}
\pi_{\theta}(\tau) = p(s_1) \prod_{t=1}^T \pi_{\theta}(a_t \vert s_t)p(s_{t+1}\vert s_t, a_t)
\end{equation}
and therefore, taking the $$\log$$ we obtain
\begin{equation}
\label{eq:log_pi_tau}
\log \pi_{\theta}(\tau) = \log p(s_1) +
\sum_{t=1}^T \log\pi_{\theta}(a_t \vert s_t) + \sum_{t=1}^T \log p(s_{t+1} \vert s_t, a_t)
\end{equation}
and since only $$\pi_{\theta}$$ depends on $\theta$, when we take the gradient we are left with
\begin{equation}
\nabla_{\theta} \log\pi_{\theta}(\tau) =
\sum_{t=1}^T \nabla_{\theta}\log\pi_{\theta}(a_t \vert s_t)
\end{equation}

Substituting the gradient into Eq. \ref{eq:pg_grad_tau} we obtain the **Policy Gradient**
\begin{equation}
\label{eq:pg_gradient}
\nabla_{\theta}J(\theta) = E_{\tau \sim \pi_{\theta}(\tau)} \left[
\left ( \sum_{t=1}^T \nabla_{\theta}\log \pi_{\theta}(a_t \vert s_t) \right) r(\tau)
\right]
\end{equation}


### Why can we subtract a baseline?
In [Lecture 5: Policy Gradients](/lectures/lecture5), we observed that introducing a baseline
$b$ reduces the variance of the gradient in Eq. \ref{eq:pg_gradient}:

\begin{equation}
\nabla_{\theta}J(\theta) = E_{\tau \sim \pi_{\theta}(\tau)} \left[
\nabla_{\theta}\log \pi_{\theta}(\tau) \left( r(\tau) - b \right) 
\right]
\end{equation}

However, when adding a baseline we must ensure that we are not introducing a *bias*, i.e. we
want the expected value of the gradient to remain the same. Therefore, we inspect the term
$$E_{\tau \sim \pi_{\theta}(\tau)} \left[ \nabla_{\theta} \log \pi_{\theta}(\tau)b \right]$$ and
we investigate the conditions under which it is equal to zero. We expand $$\pi_{\theta}(\tau)$$
as in Eq. \ref{eq:log_pi_tau} and since the gradient is zero for all the terms not depending on 
$\theta$, we are left with
\begin{equation}
\label{eq:b_sum}
E_{\tau \sim \pi_{\theta}(\tau)} \left[ \nabla_{\theta} \log \pi_{\theta}(\tau)b \right]=
\sum_{t=1}^T E_{(s_t, a_t) \sim p(s_t, a_t)}\left[
\nabla_{\theta} \log\pi_{\theta}(a_t \vert s_t) b
\right]
\end{equation}
In which we moved the expectation inside the summation. Now, we apply the
[Law of Total Expectation](https://en.wikipedia.org/wiki/Law_of_total_expectation) to the
expectation over the joint distribution $$p(s_t, a_t) = p(s_t) \pi_{\theta}(a_t \vert s_t)$$ to
write Eq. \ref{eq:b_sum} above as
\begin{equation}
\sum_{t=1}^T E_{s_t \sim p(s_t)} \left[
E_{a_t \sim \pi_{\theta}(a_t \vert s_t)} \left[ 
\nabla_{\theta}\log\pi_{\theta}(a_t \vert s_t)\ b
\right]\right]
\end{equation}
We can now expand the expectations with their integral definition
\begin{equation}
\sum_{t=1}^T \int_{s_t} p(s_t) \left( \int_{a_t} \pi_{\theta}(a_t \vert s_t) \nabla_{\theta}
\log\pi_{\theta}(a_t \vert s_t)b\ da_t\right) ds_t
\end{equation}
and, "reversing" the log-gradient trick of Eq. \ref{eq:log_grad_trick} we obtain
\begin{equation}
\sum_{t=1}^T \int_{s_t} p(s_t) \left( \int_{a_t} \nabla_{\theta} \pi_{\theta}(a_t \vert s_t)
b\ da_t\right) ds_t
\end{equation}
Finally, we move the gradient operator outside the second integral (Leibniz Integral Rule again)
\begin{equation}
\sum_{t=1}^T \int_{s_t} p(s_t)\nabla_{\theta} \left( \int_{a_t} \pi_{\theta}(a_t \vert s_t)
b\ da_t\right) ds_t
\end{equation}

At this point we must be very careful: **if** we can move $b$ outside the integral, we obtain
\begin{equation}
\sum_{t=1}^T \int_{s_t} p(s_t) b \nabla_{\theta} \left( \int_{a_t} \pi_{\theta}(a_t \vert s_t)
\ da_t\right) ds_t = 0
\end{equation}

since $$\int_{a_t} \pi_{\theta}(a_t \vert s_t)\ da_t = 1$$ by the definition of probability, and
therefore $$\nabla_{\theta}1 = 0$$.

#### When can we do this?

We obtained this result by moving $b$ outside the inner integral over the action $$a_t$$. This
is only possible if the baseline $b$ does not depend on $a_t$. Therefore, we just proved that
an *action-independent* baseline $b$ has zero expected value on th Policy Gradient and thus it
does not introduce any bias. As explained in [Lecture 6: Actor Critic](/lectures/lecture6),
**state-dependent** baselines $$b(s_t)$$ are often exploited to reduce the Policy Gradient variance
without introducing biases.
