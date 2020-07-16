---
layout: paper
title: Soft Actor-Critic (SAC)
category: algorithm
permalink: /papers/Soft-Actor-Critic
paper-author: Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine
paper-institutions: University of California, Berkeley, USA
post-author: Oleguer Canal
paper-year: 2018
paper-link: https://arxiv.org/abs/1801.01290
Brief description that should cover:
---
<!--
Disclaimer and authorship:
This article is provided for free only for your personal informational and entertainment purposes. No commercial use of it is allowed.

Please note there might be mistakes. We would be grateful to receive (constructive) criticism if you spot any. You can reach us at: ai.campus.ai@gmail.com or directly open an issue on our github repo: https://github.com/CampusAI/CampusAI.github.io

If considering to use the text please cite the original author/s of the lecture/paper.
Furthermore, please acknowledge our work by adding a link to our website: https://campusai.github.io/ and citing our names: Oleguer Canal and Federico Taschin.
-->

## Idea

Deep RL algorithms suffer from:
 - __High sample complexity:__ Most __on-policy__ algorithms require new samples to be collected after each gradient step (too expensive in large problems). 
 - __Brittle convergence:__ To solve sample complexity, __off-policy__ algorithms aim to reuse past experience. But:
     - Its not directly feasible with conventional __policy gradient__ based methods.
     - __Value-based__ approaches (e.g. Q-learning) combined with NN give a challenge for stability (which worsens if working in continuous spaces).

To solve them, this paper implements an __off-policy__ (deal with sample complexity), __actor-critic__ (work in high-dim continuous spaces) algorithm in the __maximum entropy framework__ (enhance learning robustness).

### Maximum Entropy Framework
*"Succeed at the task, while behaving as random as possible"*:
Actor aims to maximize expected reward while also maximizing entropy:
- Wider __exploration__ discarding clearly unpromising avenues.
- Better convergence __robustness__ as it prevents premature local optima convergence.

The optimization objective then becomes:

\begin{equation}
J(\pi) = \sum_t E_{(s_t, a_t) \sim \rho_{\pi}} \left[ r(s_t, a_t) + \alpha H(\pi(\cdot | s_t)) \right]
\end{equation}

Where $\alpha$ is a "temperature" meta-parameter, which can be fixed or learned ($\alpha$ = 0 for standard RL).

**THIS POST IS UNFINISHED**
<!-- ## Contribution
 - __Stable Learning:__ Results show similar performance accross different seeds, in contrast to other off-policy methods.

## Weaknesses -->

