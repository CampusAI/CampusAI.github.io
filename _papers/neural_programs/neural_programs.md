---
layout: paper
title: Strong Generalization and Efficiency in Neural Programs
category: algorithm
permalink: /papers/neural_programs
paper-author: Yujia Li, Felix Gimeno, Pushmeet Kohli, Oriol Vinyals
paper-institutions: DeepMind
post-author: Oleguer Canal
paper-year: 2020
paper-link: https://arxiv.org/pdf/2007.03629.pdf
---
<!--
Disclaimer and authorship:
This article is provided for free only for your personal informational and entertainment purposes. No commercial use of it is allowed.

Please note there might be mistakes. We would be grateful to receive (constructive) criticism if you spot any. You can reach us at: ai.campus.ai@gmail.com or directly open an issue on our github repo: https://github.com/CampusAI/CampusAI.github.io

If considering to use the text please cite the original author/s of the lecture/paper.
Furthermore, please acknowledge our work by adding a link to our website: https://campusai.github.io/ and citing our names: Oleguer Canal and Federico Taschin.
-->

reinforcement learning improvement in efficiency over hand-coded algorithms


Contributions:
- learning neural networks for algorithmic tasks that generalize to instances of arbitrary length we tested on.
- demonstrating the implications of model interfaces on generalization and efficiency, taking as inspiration CPU instruction sets
- using imitation and reinforcement learning to optimize efficiency metrics and discover new algorithms which can outperform strong teachers.

Takes inspiration from [Neural Programmer Interpreter (NPI)](https://arxiv.org/abs/1511.06279).

[Making Neural Programming Architectures Generalize via Recursion](https://arxiv.org/abs/1704.06611)

neural controller - interface framework:
where a neural model interacts with external interfaces, like output or memory, through a range of instructions,

The neural controller executes a
sequence of such instructions until a task is solved or some termination condition is met