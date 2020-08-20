---
layout: experiment
title: "Self-learned vehicle control using PPO"
permalink: /experiments/autonomous_driving
experiment-author: Oleguer Canal, Federico Taschin
code-link: https://github.com/OleguerCanal/KTH_MA-autonomous-driving
report-link: /pdf/autonomous_driving.pdf
---
<!--
Disclaimer and authorship:
This article is provided for free only for your personal informational and entertainment purposes. No commercial use of it is allowed.

Please note there might be mistakes. We would be grateful to receive (constructive) criticism if you spot any. You can reach us at: ai.campus.ai@gmail.com or directly open an issue on our github repo: https://github.com/CampusAI/CampusAI.github.io

If considering to use the text please cite the original author/s of the lecture/paper.
Furthermore, please acknowledge our work by adding a link to our website: https://campusai.github.io/ and citing our names: Oleguer Canal and Federico Taschin.
-->

## Context
This project was done as part of a contest in [DD2438 Artificial Intelligence and Multi Agent Systems](https://www.kth.se/student/kurser/kurs/DD2438?l=en) course at [KTH](https://www.kth.se/en).


## Abstract

This work tackles the completion of an obstacle maze by a self-driving vehicle.
We solve it combining two main ideas:
- First, we plan an approximate path running Dijkstra on the environment's Visibility Graph.
- Second, a fully self-trained agent using PPO (Proximal Policy Optimization) controls the vehicle making it follow the pre-computed path the fastest way possible.

Results show a high degree of environment generalization achieved by training on randomized maps of increasing difficulty (Curriculum Learning).
Furthermore, our data-driven control approach usually outperforms any of the other heuristic-based methods attempted in both maze completion time and natural driving feel, making us the team with lowest summed time added over all test tracks.

## Environment

## Path planning

## Control learning

## Results