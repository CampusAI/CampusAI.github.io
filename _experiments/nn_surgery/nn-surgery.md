---
layout: experiment
title: "Neural Network Surgery in Deep Reinforcement Learning"
permalink: /experiments/nn_surgery
experiment-author: Oleguer Canal, Federico Taschin
experiment-date: May 2020
code-link: https://github.com/CampusAI/NNSurgery
report-link: /pdf/
---
<!--
Disclaimer and authorship:
This article is provided for free only for your personal informational and entertainment purposes. No commercial use of it is allowed.

Please note there might be mistakes. We would be grateful to receive (constructive) criticism if you spot any. You can reach us at: ai.campus.ai@gmail.com or directly open an issue on our github repo: https://github.com/CampusAI/CampusAI.github.io

If considering to use the text please cite the original author/s of the lecture/paper.
Furthermore, please acknowledge our work by adding a link to our website: https://campusai.github.io/ and citing our names: Oleguer Canal and Federico Taschin.
-->

## Context
As we learned in many of our [experiments](/experiments), training Deep Reinforcement Learning agents
can be an extremely time consuming task. Moreover, it is often the case that one needs to change the
setup. As an example, in our [Autonomous Driving](/experiments/autonomous_driving), we often made
changes to the observation space, such as changinf the number of lidar rays and next visible path
points. In other cases one may want to add a new action to an agent that was previously trained, or
to modify the size and shape of the neural network. All these changes to the network architecture
require the agent to be trained again. In this work, taking inspiration from
[Neural Network Surgery with Sets](https://arxiv.org/abs/1912.06719), we implement a simple weight
transplanting functionality that allows us to move weights from a trained network $$\theta^{old}$$ to
a new network $$\theta^{new}$$ with a different architecture, transfering the old weights in the
appropriate place and initializing the new ones.

## Approach

## Environment

### Car action-space

### State-space

### Reward system

## Control learning

## Results

## Future work

## Takeaways
