---
layout: lecture
title: "Lecture 11: Model-based RL"
permalink: /lectures/lecture11
lecture-author: Sergey Levine
lecture-date: 2019
slides-link: http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-11.pdf
video-link: https://www.youtube.com/watch?v=pE0GUFs-EHI&list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A&index=12&
---

**Idea:** If we learn $$f(s_t, a_t) = s_{t+1}$$ (or $$p(s_{t+1} \mid s_t, a_t)$$ in stochastic envs.) we can apply [last lecture](/lectures/lecture10) techniques.

# Naive approaches

### System identification in classical robotics:
{% include figure.html url="/_lectures/lecture_11/si_1.png"%}

**Does it work?**
Yes, if we design a good base policy and we hand hand-engineer a dynamics representation of the environment (using physics knowledge) and we just fit few parameters.

**Distribution mismatch problem:** When we design a policy, we might be extrapolating beyond the data distribution used to learn the physics model, making results to be far off.

**Solution:** (Inspired by DAgger) Add samples from estimated optimal trajectory.

{% include figure.html url="/_lectures/lecture_11/si_2.png"%}

**Problem:** Blindly planning on top of an imperfect learned env. model might still result into ending up taking actions in states we did not plan to be. To improve this, we can update the plan after each step:

{% include figure.html url="/_lectures/lecture_11/si_3.png"%}

**OBS:** This algorithm works very decently: Replanning after each step compensates for imperfections in model and in planner. We can even plan for shorter horizons or use some very simple planner like random shooting.

**Problems:**
- Replanning after every step can be very expensive if we use good planners.
- Model uncertainty may lead into overestimating regions resulting into the algorithm not progressing.

# Uncertainty in model-based RL

**Idea:** What if we exploit models being aware of their uncertainty? (similarly to Gaussian Processes)

There exist 2 types of uncertainty:
- The **aleatoric** (or statistical) uncertainty: Regarding the samples of the data. A good model of high aleatoric uncertainty data can have relative high entropy.
- The **epistemic** (or model) uncertainty: When the model is certain about the data, but we are not certain about the model (e.g. overfiting). You cannot infer that uncertainty from output entropy.

**OBS:** We cannot just use entropy of the ANN outcome since it is not a good measure of uncertainty. The model could be overfited and be very confident about something it is wrong about.

## Estimating the posterior: $$p(\theta \mid \mathcal{D})$$

When training models, we usually only get a Maximum Likelihood Estimation (MLE) of the params: $$argmax_\theta \log p(\theta \mid \mathcal{D})$$ (which if the priors are uniform is the same as $$argmax_\theta \log p(\mathcal{D} \mid \theta)$$).\\
Nevertheless, $$p(\theta \mid \mathcal{D})$$ would tell us the real uncertainty of the model. Moreover, we could make predictions marginalizing over the parameters:

\begin{equation}
p(s_{t+1} \mid s_t, a_t) = \int p(s_{t+1} \mid s_t, a_t, \theta) p(\theta \mid \mathcal{D}) d\theta
\end{equation}

### Bootstrap ensembling
We can achieve a rough approximation of $$p(s_{t+1} \mid s_t, a_t)$$ by training multiple independent models and averaging them as:

\begin{equation}
\label{bootstrap}
\int p(s_{t+1} \mid s_t, a_t, \theta) p(\theta \mid \mathcal{D}) d\theta \simeq
\frac{1}{N} \sum_i p(s_{t+1} \mid s_t, a_t, \theta_i)
\end{equation}

Bootstrap ensembling achieves the training of "independent" models by  training on "independent" datasets:
It uses **bootstrap samples**, uniform sampling with replacement of the entire dataset.
Later, to get an estimation of $$p(\theta \mid \mathcal{D})$$ we can just average each model output as depicted in Eq. \ref{bootstrap}.

**OBS:** Bootstrap sampling is not very needed since SGD and random initialization is usually sufficient to make models independent enough. 

**Problem:** Training NNs is expensive so usually the number of fitted models is very small (< 10), this makes the approximation to be very crude.

**Solution:** In later lectures we will present better methods such as the use of **Bayesian NN (BNN)**, **Mean Field approximations** and **variational inference** in general.

### Planning with uncertainty

#### Deterministic environments

[Before](/lectures/lecture10) a single model predicted next state:

\begin{equation}
J(a_1,...,a_T) = \sum_t r(s_t, a_t)
\space \space \space \space s.t. \space 
s_{t+1} = f(s_t, a_t)
\end{equation}

Now, the next states are predicted by an average of multiple models:
\begin{equation}
J(a_1,...,a_T) = \frac{1}{N} \sum_i \sum_t r(s_{t, i}, a_t)
\space \space \space \space s.t. \space 
s_{t+1, i} = f_i (s_{t, i}, a_t)
\end{equation}

#### Stochastic environments:

In case of a stochastic environment, the algorithm becomes:

{% include figure.html url="/_lectures/lecture_11/stoc_bootstrap.png"%}

**OBS:** There exist fancier options such as **moment matching** or better posterior estimation with **BNNs**.


# Model-based RL with complex observations

**Problems** in estimating $$f(s_t, a_t) = s_{t+1}$$ big observation spaces such as images:
- **High dimensionality** makes it harder to fit an environment model.
- **Redundancies**: Some pixel values behave the same way even if far apart because of intrinsic correlations.
- **Partial observability**: Images tend to be a partial observation of the underlying state.

Remember that states are the underlying structure which fully describe an instance of the environment, while observations are only what the agent perceives.\\
**Example:** in an Atari game, the observation is the screen output, and the state can be summarized in less dimensions by the position of the key elements.

**Idea:** Can we learn separately $$p(o_t \mid s_t )$$ and $$p(s_{t+1} \mid s_t, a_t)$$
- **$$p(o_t \mid s_t )$$**: Is high-dimensional but static.
- **$$p(s_{t+1} \mid s_t, a_t)$$**: Is low-dimensional but dynamic.

### Latent space models

### Observations model