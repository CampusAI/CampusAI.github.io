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

{% include start-row.html %}

<blockquote markdown="1">
**Attention** is the ability to focus on what is important for the desired task. Contrary to memory, its not about retaining as much as possible, but to discard what is not needed.
</blockquote>

## Implicit attention is not enough!

Common ANNs already encode some sort of attention: they learn on what to focus on given some inputs.

- **Feed-forward networks:** A very visual example can be seen in our [GLOW post](/papers/Grad-CAM), a method to highlight the most relevant areas of an image to its classification.
To see how important each input field is wrt to each label, we can compute the Jacobian of the ANN: $$\frac{\partial label}{\partial input}$$

{% include end-row.html %}
{% include start-row.html %}
- **Recurrent networks:** Similarly, we can see seq2seq RNNs' most sensitive input signal by analyzing their **Sequential Jacobian**: $$J_k^t = \left( \frac{\partial y_k^t}{\partial x^1}, \frac{\partial y_k^t}{\partial x^2}, ... \right)$$. For each  label, it gives us how "relevant" each sequence input is.

Despite the merit of these results, designing an architecture with attention mechanisms has notable advantages:
- Higher computational **efficiency**
- Greater **scalability**: Usually size of the input does not affect the size of the architecture.
- **Sequential processing** of static data: We can just remember the important information of previous data.
- Easier **interpretation**: As they are specifically designed for the purpose of attention.

{% include annotation.html %}
[Neural Machine Translation in Real time](https://arxiv.org/abs/1610.10099) paper shows how a deep-enough seq2seq RNN without any explicit attention mechanism is able to efficiently learn the different word orderings.
{% include end-row.html %}
{% include start-row.html %}

## Neural Attention Models

ddsd

## dskjsjkdfjs

dededes


{% include end-row.html %}
