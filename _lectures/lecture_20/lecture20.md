---
layout: lecture
title: "Lecture 20: Meta-Reinforcement Learning"
permalink: /lectures/lecture20
lecture-author: Kate Rakelly
lecture-date: 2019
slides-link: http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-20.pdf
video-link: https://www.youtube.com/watch?v=4qH_h5_V3O4&list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A&index=20&t=7s
---
<!--
Disclaimer and authorship:
This article is provided for free only for your personal informational and entertainment purposes. No commercial use of it is allowed.

Please note there might be mistakes. We would be grateful to receive (constructive) criticism if you spot any. You can reach us at: ai.campus.ai@gmail.com or directly open an issue on our github repo: https://github.com/CampusAI/CampusAI.github.io

If considering to use the text please cite the original author/s of the lecture/paper.
Furthermore, please acknowledge our work by adding a link to our website: https://campusai.github.io/ and citing our names: Oleguer Canal and Federico Taschin.
-->

### Regular RL
Learn an optimal policy (optimal action given a state) for a single task. I.e.
fit network parameters for a given MDP:

\begin{equation}
\theta^\star = argmax_\theta E_{\pi_\theta (\tau)} \[ R( \tau ) \]
\end{equation}

### Meta-RL
Learn adaptation procedure.
If we have a set of $$MDPs = \{ MDP_1, ... MDP_n \}$$ that share some common structure, we can learn some common parameters $$\theta$$.\\
This algorithm step is known as **Meta-training** or **Outer loop**:

\begin{equation}
\theta^\star = argmax_\theta \sum_i E_{\pi_{\phi_i} (\tau)} \[ R( \tau ) \]
\end{equation}

From those, we can fit any individual related task with few samples.
This means that for $$MDP_i$$, we will derive its optimal policy parameters $$\phi_i$$ from the meta-learned ones.\\
This algorithm step is known as **Adaptation** or **Inner loop**:

\begin{equation}
\phi_i = f_\theta (MDP_i)
\end{equation}

The adaptation procedure has 2 main goals:
- **Exploration**: Collect most informative data
- **Adaptation**: Use that data to obtain the optimal policy.

**Notice**: The adaptation policy (defined by params $$\theta$$) does not need to be good at any task, only to be easily adaptable to all of them. We can think of $$\theta$$ as a prior we'll use to learn a posterior for each task.

## Meta-RL algorithms:
The most basic algorithm idea we can try is:

While training:
  1. Sample task $$i$$, collect data $$\mathcal{D}_i$$
  2. Adapt policy by computing: $$\phi_i = f(\theta, \mathcal{D}_i)$$
  3. Collect data $$\mathcal{D}_i^\prime$$ using adapted policy $$\pi_{\phi_i}$$
  4. Update $$\theta$$ according to $$\mathcal{L} (D_i^\prime, \phi_i)$$

**Notice**: Steps 1-3 belong to the **adaptation** step while step 4 belongs to the **meta-training** step.

**Improvements**:
- Run multiple rounds of the adaptation step
- Compute $$\theta$$ update (step 4) across a batch of tasks.

From now on, we'll mainly be discussing different choices for function $$f$$ and loss $$\mathcal{L}$$.

### Recurrence algorithm
We want to collect new information while recalling what we've seen so far.
Thus, it seems that encoding the policy as a **Recurrent ANN** can be a good idea.
The agent will collect $$(s, a, r)$$ tuples as it interacts with the environment and use them to update its internal state.

The internal state is individual to each task but kept constant between different episodes of the same task as this image depicts:

{% include figure.html url="/_lectures/lecture_20/recurrent_alg_idea.png" description="Recurrent algorithm idea" %}

Then the basic algorithm becomes:

{% include figure.html url="/_lectures/lecture_20/recurrent_alg.png" description="Recurrent algorithm pseudocode." %}

**Pros/Cons**:
+ <span style="color:green">It is **general** and **expressive**: There exists and RNN that can compute any function.</span>
+ <span style="color:red">It is **NOT consistent**: There is no guarantee it will converge to the optimal policy.</span>

More on: [Benchmarking Deep Reinforcement Learning for Continuous Control](https://arxiv.org/abs/1604.06778) and [Learning to reinforcement learn](https://arxiv.org/abs/1611.05763).

### Optimization algorithm

**Idea**: Learn a parameter initialization from which fine-tunning for a new task works easily.

{% include figure.html url="/_lectures/lecture_20/optimization_idea_2.png" description="Optimization algorithm idea" %}

The algorithm can be written as:

{% include figure.html url="/_lectures/lecture_20/optimization_alg.png" description="Optimization algorithm idea" %}

**Notice**: $$\theta$$ receives credit for providing good exploration policies.

**Pros/Cons**:
+ <span style="color:green">It is **consistent**: It is just gradient descent.</span>
+ <span style="color:red">It is **NOT as expressive**: If no rewards are collected adaptation wil not change the policy, even when this data gives information about states to avoid.</span>

More on: [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400) and [ProMP: Proximal Meta-Policy Search](https://arxiv.org/abs/1810.06784).

## Meta-imitation learning

Robot RL example