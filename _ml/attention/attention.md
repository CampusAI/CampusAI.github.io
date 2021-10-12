---
layout: article
title: "Attention basics"
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

Common ANNs already encode some sort of **implicit attention**: they learn on what to focus on, given some inputs.

- **Feed-forward networks:** A very visual example can be seen in our [GLOW post](/papers/Grad-CAM): a method to highlight the most relevant areas of an image for a particular label.
This is done computing the Jacobian of the ANN wrt each input pixel: $$\frac{\partial label}{\partial input}$$ (i.e. how much of an impact has each pixel to the label).

{% include end-row.html %}
{% include start-row.html %}
- **Recurrent networks:** Similarly, we can see seq2seq RNNs' most sensitive input signal by analyzing their **Sequential Jacobian**: $$J_k^t = \left( \frac{\partial y_k^t}{\partial x^1}, \frac{\partial y_k^t}{\partial x^2}, ... \right)$$. For each output (at a particular time), it gives us how "relevant" each element of the inputted sequence is.

Despite the merit of these results, designing an architecture with attention mechanisms has notable advantages:
- Higher computational **efficiency**
- Greater **scalability**: Usually size of the input does not affect the size of the architecture.
- **Sequential processing** of static data: We can just remember the important information of previous data.
- **Easier interpretation**: As they are specifically designed for the purpose of attention.

{% include annotation.html %}
[Neural Machine Translation in Real time](https://arxiv.org/abs/1610.10099) paper shows how a deep-enough seq2seq RNN without any explicit attention mechanism is able to efficiently learn the different word orderings in different languages.
{% include end-row.html %}
{% include start-row.html %}

## Neural Attention Models

### Hard (non-differentiable) attention

{% include end-row.html %}
{% include start-row.html %}

The model operates sequentially and is composed by two components:

- An **attention model**:
  - Input: Original data, information vector
  - Output: Fixed-size "glimpse" (representation of a partition of the input / part of the input).
- The **network** itself:
  - Input: Data "glimpse"
  - Output: Both the desired output and an extra vector used in the attention model

Usually, the **attention model** works by giving a probability distribution over "glimpses" $$g$$ of the original data $$x$$ given some set of attention outputs $$\vec{a}$$.

<blockquote markdown="1">
**For instance:** If the input is an image, the glimpses can be different tiles of a partition of it.

{% include figure.html url="/_ml/attention/glimpse_distribution.png" description="Glimpse example of an image input. (Image from DeepMind)" width="30" zoom="1.0"%}

**Notice** that this is just an example, good-performing models have much more complex glimpse systems. For instance, combining resizings of different resolutions of different parts of the image.
</blockquote>

{% include annotation.html %}
{% include figure.html url="/_ml/attention/neural_attention_models.png" description="Neural Attention Model architecture. (Image from DeepMind)" width="40" zoom="1.0"%}
{% include end-row.html %}
{% include start-row.html %}

So, given an attention vector $$\vec{a}$$, we have a probability distribution

\begin{equation}
P(g_k \mid \vec{a})
\end{equation}

Thinking in reinforcement learning terms, this can be understood as a **stochastic policy** $$\pi_a$$: the attention model needs to choose which glimpse to provide to the network.

However, this is a **hard decision**: we no longer have a gradient over the input, we must choose a particular "glimpse" (take a sample of the distribution).
Luckily, to train this "policy" $$\pi$$, we can rely on [policy-gradient](/lectures/lecture5) RL techniques such as the REINFORCE algorithm (using the loss of the main network as a reward signal).

{% include end-row.html %}
{% include start-row.html %}

{% include figure.html url="/_ml/attention/hard_attention.gif" description="Cluttered MINST classification using hard attention: The network gets inputed glimpses which it chooses.  (Image from [Aryan Mobiny](https://github.com/amobiny/Recurrent_Attention_Model))" width="60" zoom="1.0"%}

This results in a high **scalable** solution: The model changes focus on different parts of images of different sizes by changing the area of interest while deciding what to output. (mimicking the movement of a human eye over an image)

{% include annotation.html %}
More on [Recurrent Visual Attention](https://github.com/kevinzakka/recurrent-visual-attention)
{% include end-row.html %}
{% include start-row.html %}

### Soft (differentiable) attention

Soft attention methods do not give explicit attention but allow end-to-end backprop training.

The most basic approach would be to take the **expectation** (instead of a sample) of the "glimpse" distribution $$P(g_k \mid a)$$ presented before:

{% include end-row.html %}
{% include start-row.html %}
\begin{equation}
g = \sum_{g^\prime \in X} g^\prime \cdot P (g^\prime \mid \vec{a})
\end{equation}

{% include annotation.html %}
This is differentiable wrt $$\vec{a}$$ [if $$P(g \mid \vec{a})$$ is]
{% include end-row.html %}
{% include start-row.html %}

**Attention weights**:
Notice that we don't really need a distribution though.
A set of weights $$\{ w_i \}_i$$ over the "glimpses" can be used to define an **attention readout** $$\vec{v}$$ from some representation (meaningful embedding) of the "glimpses" $$\vec{v_i}$$:

{% include end-row.html %}
{% include start-row.html %}

\begin{equation}
\vec{v} = \sum_i w_i \vec{v_i}
\end{equation}

{% include annotation.html %}
Not needed, but usually it is nice that $$\sum_i w_i = 1$$ and $$w_i \in [0, 1] \forall i$$
{% include end-row.html %}
{% include start-row.html %}

{% include end-row.html %}
{% include start-row.html %}

Now you might be thinking something like: *"Wait, this is very similar to linear layer weights..." And they are similar! However:

- Attention weights change dynamically with the input sequence (they are data-dependent, aka **fast weights**)
- Common ANN weights do NOT directly depend on the input data at inference time. They change "slowerly" with gradient steps in training time.

{% include annotation.html %}
For instance, we can think of Conv1D's layer operation as applying a set of weights on a set of "glimpses" of the input vector:

{% include figure.html url="/_ml/attention/conv1d.gif" description="Conv1d mechanism. (Image from [krzjoa](https://krzjoa.github.io/))" width="80" zoom="1.0"%}

However, the kernel is fixed (both values and size) regardless fo the input!

{% include end-row.html %}
{% include start-row.html %}





## dskjsjkdfjs

dededes


{% include end-row.html %}
