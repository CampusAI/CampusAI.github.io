---
layout: page
title: Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition
permalink: /papers/HRL_with_MAXQ_decomposition
---
Hierarchical Reinforcement Learning aims to decompose the target Markov Decision Process into a
combination of smaller MDPs with different levels of abstraction. This paper presents a new approach,
the MAXQ decomposition, to decompose the Value Function for a given hierarchical policy. It formally
defines the MAXQ hierarchy, provides conditions for using state abstractions, and a model-free
online learning algorithm that converges to a Recursively Optimal Policy.

# Idea
- **Define a hierarchical abstraction of how the agent should behave:** The programmer must decompose the
target MDP into smaller MDPs with different levels of abstraction. One example is the Taxi Navigation problem
in Figure 1.

{% include figure.html url="/assets/images/HRL_with_MAXQ_decomposition/taxi_navigation.png" description="Figure 1: Taxi navigation problem decomposed into smaller sub-problems" %}
