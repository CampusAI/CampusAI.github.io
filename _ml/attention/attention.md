---
layout: article
title: "Attention"
permalink: /ml/attention
content-origin: Alex Graves Attention and Memory in Deep Learning lecture, lilianweng.github.io
post-author: Oleguer Canal
---
<!--
Disclaimer and authorship:
This article is provided for free only for your personal informational and entertainment purposes. No commercial use of it is allowed.

Please note there might be mistakes. We would be grateful to receive (constructive) criticism if you spot any. You can reach us at: ai.campus.ai@gmail.com or directly open an issue on our github repo: https://github.com/CampusAI/CampusAI.github.io

If considering to use the text please cite the original author/s of the lecture/paper.
Furthermore, please acknowledge our work by adding a link to our website: https://campusai.github.io/ and citing our names: Oleguer Canal and Federico Taschin.
-->

- **Attention**: Ability to focus on what is important for the desired task. Contrary to memory, its not about retaining as much as possible, but to discard what is not needed.

## Attention in FFNNs and RNNs

ANNs already encode some sort of attention: learn on what to focus on given some inputs.

**Feed-forward networks:** A very visual example can be seen in our [GLOW post](/papers/Grad-CAM), a method to highlight the most relevant areas of an image to its classification.
To see how important each input field is, we can see the Jacobian of the ANN: $$\frac{\partial label}{\partial input}$$

**Recurrent networks:** Similarly, we can see RNNs' most sensitive input signal by analyzing their **Sequential Jacobian**.