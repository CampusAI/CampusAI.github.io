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

_Ok, I've dragged it enough... Time has come to see whats behind all this transformer hype :)
This post explores the concept of attention in ANNs.
In later ones, we'll see how we can leverage these ideas in common ML tasks.
But first, lets start with a definition:_

<blockquote markdown="1">
**Attention** is the ability to focus on what is important for the desired task. Contrary to memory, its not about retaining as much as possible, but to discard what is not needed.
</blockquote>

## Implicit attention is not enough!

But wait, aren't regular ANNs already acquiring some sense of attention?
Good question! And yes: provided some inputs, common ANNs learn on what to focus on.
This is known as **implicit attention** and some examples are:

- **Feed-forward networks:** A very visual example can be seen in our [GLOW post](/papers/Grad-CAM): a method to highlight the most relevant areas of an image for a particular label.
This is done computing the Jacobian of the ANN wrt each input pixel: $$\frac{\partial label}{\partial input}$$ (i.e. how much of an impact has each pixel to the label).

{% include end-row.html %}
{% include start-row.html %}
- **Recurrent networks:** Similarly, we can see seq2seq RNNs' most sensitive input signal by analyzing their **Sequential Jacobian**: $$J_k^t = \left( \frac{\partial y_k^t}{\partial x^1}, \frac{\partial y_k^t}{\partial x^2}, ... \right)$$. For each output (at a particular time), it gives us how "relevant" each element of the inputted sequence is.

Despite the merit of these results, designing an architecture with explicit attention mechanisms has notable advantages:
- Higher computational **efficiency**
- Greater **scalability**: Usually size of the input does not affect the size of the architecture.
- **Sequential processing** of static data: We can just remember the important information of previous data.
- **Easier interpretation**: As they are specifically designed for the purpose of attention.

Sounds like a good deal! So let's see how we can design models which explicitly use attention mechanisms.

{% include annotation.html %}
[Neural Machine Translation in Real time](https://arxiv.org/abs/1610.10099) paper shows how a deep-enough seq2seq RNN without any explicit attention mechanism is able to efficiently learn the different word orderings in different languages.
{% include end-row.html %}
{% include start-row.html %}

## Neural Attention Models

{% include end-row.html %}
{% include start-row.html %}

To understand how most attention-based models operate, it is key to first understand the concept of glimpses.

<blockquote markdown="1">
**Glimpses** are representations of partitions of the input data.
For instance the glimpses of a:
- **Text input** might be an array of 1-hot encoding of its letters or an array of its word embeddings.
- **Sound input** might be a wave representation of the sound at fixed periods of time.
- **Image input** might be different tiles of partitions of the image (see left-side note).
- **Video input** might be the different images which compose the video.
</blockquote>

Attention-based models are composed by 2 modules which operate in a cyclic fashion (even for static data):
- An **attention mechanism**: Chooses which glimpses (or what glimpse combination) to input to the ANN.
- The **main ANN**: Produces the desired output and provides feedback to the attention mechanism.

The iteration loop looks as follows:
1. The network provides some _"attention vector"_ to the attention mechanism.
2. The attention mechanism "chooses" what to input to the network (which is dependent on this _"attention vector"_).
3. The network processes the input, generates an output (and/or hidden state).
4. Go back to 1.
{% include figure.html url="/_ml/attention/neural_attention_models.png" description="Neural Attention Model architecture. (Image from DeepMind)" width="40" zoom="1.0"%}

Thus, what defines an attention mechanism **is the way the glimpses of original data are chosen/combined according to the _"attention vector"_**.
There are a lot of ways to do that, so in this post we'll focus on some of the most influential works.

{% include annotation.html %}
{% include figure.html url="/_ml/attention/glimpse_distribution.png" description="Glimpse example of an image input. (Image from DeepMind)" width="50" zoom="1.0"%}
**Notice** that this is just an example, good-performing models have much more complex glimpse systems. For instance, combining resizings of different resolutions of different parts of the image.
{% include end-row.html %}
{% include start-row.html %}


### Hard Attention

{% include end-row.html %}
{% include start-row.html %}
The hard attention mechanism (aka non-differentiable attention) is characterized by providing a single glimpse $$g$$ to the network at each iteration.
Internally, it works by building a probability distribution over glimpses $$\{ g_k \}_k$$ of the original data $$x$$ given some set of attention outputs $$\vec{a}$$:

\begin{equation}
P(g_k \mid \vec{a})
\end{equation}

Notice that (if thinking in reinforcement learning terms) this can be understood as a **stochastic policy** $$\pi_a$$: the attention model needs to choose which glimpse to provide to the network at each iteration.

{% include annotation.html %}
In the image example, this probability distribution would be a categorical distribution which assigns a probability to each glimpse.
These probabilities depend on what the network wants to see next:
maybe it has detected that there is something interesting to the left of the previously seen glimpse, so it will give a higher probability to the glimpses at the left.
{% include end-row.html %}
{% include start-row.html %}

This is a **hard decision**: we no longer have a gradient over the input, we must choose a particular glimpse (take a sample of the distribution).
Which prevents end-to-end training. 
Luckily, to train this "policy" $$\pi$$, we can rely on [policy-gradient](/lectures/lecture5) RL techniques such as the REINFORCE algorithm (using the loss of the main network as a reward signal).

{% include end-row.html %}
{% include start-row.html %}

{% include figure.html url="/_ml/attention/hard_attention.gif" description="Cluttered MINST classification using hard attention: The network gets inputed glimpses which it chooses.  (Image from [Aryan Mobiny](https://github.com/amobiny/Recurrent_Attention_Model))" width="60" zoom="1.0"%}

This results in a high **scalable** solution:
The model changes focus on different parts of images while deciding what to output. (mimicking the movement of a human eye over an image)

{% include annotation.html %}
More on [Recurrent Visual Attention](https://github.com/kevinzakka/recurrent-visual-attention)
{% include end-row.html %}
{% include start-row.html %}

### Soft Attention

Soft attention methods (aka differentiable attention methods)
allow end-to-end backprop training by smoothly combining all glimpses $$\{ g_i \}_i$$.
A common approach is to take the **expectation** (instead of a sample) of the glimpse distribution $$P(g_k \mid a)$$ presented before:

{% include end-row.html %}
{% include start-row.html %}
\begin{equation}
g = \sum_{g^\prime \in X} g^\prime \cdot P (g^\prime \mid \vec{a})
\end{equation}

{% include annotation.html %}
This is differentiable wrt $$\vec{a}$$ [if $$P(g \mid \vec{a})$$ is]
{% include end-row.html %}
{% include start-row.html %}

This might sound fancy but it is super simple:
_we are only linearly combining all of the glimpses weighting each one by some factor._
In fact, notice that we don't even need to define a distribution, we can substitute these "probabilities" by a set of weights $$\{ w_i \}_i$$.
These are known as attention weights and play a key role in a lot of attention mechanisms:

{% include end-row.html %}
{% include start-row.html %}

<blockquote markdown="1">
**Attention weights**:
A set of weights $$\{ w_i \}_i$$ over the glimpses $$\{ g_i \}_i$$ can be used to define an **attention readout** $$\vec{v}$$ from some representation (meaningful embedding) of the glimpses $$\{ \vec{v_i} \}_i$$:

\begin{equation}
\vec{v} = \sum_i w_i \vec{v_i}
\end{equation}

</blockquote>
{% include annotation.html %}
Not needed, but usually it is nice that $$\sum_i w_i = 1$$ and $$w_i \in [0, 1] \forall i$$
{% include end-row.html %}
{% include start-row.html %}

{% include end-row.html %}
{% include start-row.html %}

Now you might be thinking something like:
*"Wait, how is that different from linear layer weights..."*
Another good question!
They are quite similar, however:

- Attention-based models operate in a cyclic fashion on a given input (unlike standard linear layer weights which all the data does a single forward pass)
- Attention weights change dynamically with the input sequence (they are data-dependent, aka **fast weights**)
- Common ANN weights do NOT directly depend on the input data at inference time. They change "slowerly" with gradient steps in training time.

{% include annotation.html %}
For instance, we can think of Conv1D's layer operation as applying a set of weights on a set of glimpses of the input vector:

{% include figure.html url="/_ml/attention/conv1d.gif" description="Conv1d mechanism. (Image from [krzjoa](https://krzjoa.github.io/))" width="80" zoom="1.0"%}

However, the kernel is fixed (both values and size) regardless fo the input!
The same operation is applied to all the glimpses.
Attention weights, however, would linearly combine all the glimpses of the vector to produce an output. 
{% include end-row.html %}
{% include start-row.html %}



### Associative Attention

{% include end-row.html %}
{% include start-row.html %}

Until now, all the attention mechanisms described focus on **where** to attend in the input (location-based attention).
Instead of choosing where to look, **associative attention** (aka content-based attention) attends to the content it wants to look at.
Thus, the attention parameter created by the network does not request position but content.

{% include annotation.html %}
Associative attention is currently one of the most commonly used attention mechanisms
{% include end-row.html %}
{% include start-row.html %}

In associative attention, the attention parameter created by the network is a **key vector** $$\vec{k}$$ which is compared to all the elements in the input data $$\{ \vec{x_i} \}_i$$ using some similarity function $$S(\cdot, \cdot)$$  . This gives a set of attention weights $$\{ \vec{w_i} \}_i$$ computed as:

{% include end-row.html %}
{% include start-row.html %}

\begin{equation}
w_i = \text{SOFTMAX} (S(\vec{k}, \vec{x_i}))
\end{equation}

Then, the input to the network (aka attention readout) becomes the weighted average of all the glimpses by their associated weight:

{% include annotation.html %}
$$S(\cdot, \cdot)$$ can either be fixed (e.g. dot product, cosine similarity...) or learned (e.g. MLP, Linear Operator...)
{% include end-row.html %}
{% include start-row.html %}

\begin{equation}
\text{INPUT} = \sum_i w_i x_i
\end{equation}

{% include annotation.html %}
Basically you are averaging the input partitions by _"how much you care"_ about that partition at that particular point.
{% include end-row.html %}
{% include start-row.html %}

{% include end-row.html %}
{% include start-row.html %}

Often, instead of using the glimpse $$\{ x_i \}_i$$ itself, each glimpse is expressed by a key-value pair: $$\{k_i, v_i\}$$.
Where $$k_i$$ is used to define the attention weights, and $$v_i$$ to define the value readout:

\begin{equation}
w_i = \text{SOFTMAX} (S(\vec{k}, \vec{k_i}))
\end{equation}

\begin{equation}
\text{INPUT} = \sum_i w_i v_i
\end{equation}

{% include annotation.html %}
This is useful to achieve a separation between what is used to lookup the data and what you actually get back when reading it out.
{% include end-row.html %}
{% include start-row.html %}

### Introspective attention

{% include end-row.html %}
{% include start-row.html %}

So far we have seen attention mechanisms applied to the external data being fed to the network.
Instead, **introspective attention** brings the concept of attention to the internal state of the network (aka memory).

{% include annotation.html %}
Pick up thoughts or memories.
"Memory is attention through time"
{% include end-row.html %}
{% include start-row.html %}

Left out: [Differentiable Visual Attention](https://arxiv.org/abs/1502.04623)

## Summary



{% include end-row.html %}
