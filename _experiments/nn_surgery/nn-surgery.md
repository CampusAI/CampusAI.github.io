---
layout: experiment
title: "Neural Network Surgery in Deep Reinforcement Learning"
permalink: /experiments/nn_surgery
experiment-author: Oleguer Canal, Federico Taschin
experiment-date: May 2020
code-link: https://github.com/CampusAI/NNSurgery
report-link: /pdf/nn-surgery-report.pdf
---
<!--
Disclaimer and authorship:
This article is provided for free only for your personal informational and entertainment purposes. No commercial use of it is allowed.

Please note there might be mistakes. We would be grateful to receive (constructive) criticism if you spot any. You can reach us at: ai.campus.ai@gmail.com or directly open an issue on our github repo: https://github.com/CampusAI/CampusAI.github.io

If considering to use the text please cite the original author/s of the lecture/paper.
Furthermore, please acknowledge our work by adding a link to our website: https://campusai.github.io/ and citing our names: Oleguer Canal and Federico Taschin.
-->
<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/HD8ujXtwo8A"
frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
allowfullscreen></iframe>
</div>

## Context
As we learned in many of our [experiments](/experiments), training Deep Reinforcement Learning agents
can be an extremely time consuming task. Moreover, it is often the case that one needs to change the
setup. As an example, in our [Autonomous Driving](/experiments/autonomous_driving), we often made
changes to the observation space, such as changing the number of lidar rays and next visible path
points. In other cases one may want to add a new action to an agent that was previously trained, or
to modify the size and shape of the neural network. All these changes to the network architecture
require the agent to be trained again. In this work, taking inspiration from
[Neural Network Surgery with Sets](https://arxiv.org/abs/1912.06719), we implement a simple weight
transplanting functionality that allows us to move weights from a trained network to a new network 
with a different architecture, transfering the old weights in the appropriate place and initializing
the new ones. With our extensive experiments we show that this simple weight transplant **achieves 
faster and better training** if compared with training the new network from scratch.


## Change the paradigm!
The situation in which many Reinforcement Learning researchers or practitioners may find themselves
into falls in the pattern:

1. Choose the observation space for the given problem, the action space, the network size and
   parameters
2. Train the agent
3. Realize that the action space or the observation space was not good, the network was too big/small
4. Modify the network architecture by adding/removing inputs, outputs, layers
5. Train again from scratch

Our project **aims to change this paradigm** into

1. Choose the observation space for the given problem, the action space, the network size and
   parameters
2. Train the agent
3. Realize that the action space or the observation space was not good, the network was too big/small
4. Modify the network architecture by adding/removing inputs, outputs, layers
5. **Transplant weights from the old, trained network to the new one**
6. Train the new network
7. Achieve **faster** and **better** training results!

## Transplanting the weights

### Labeling layers and units
Transplanting the weights is pretty straightforward. We use IDs to identify layers and input/output
units. Therefore, for the **old network**, the input layer units will have ids
$$I^{old} = (i_1, i_2, \: ...\: i_n)$$, the output layer units will have ids
$$O^{old} = (o1, o2, \: ... \: o_m)$$, and layers are identified by
$$L^{old} = (l_1, l_2, \: ... \: l_k)$$. When we create a new architecture and we want to transplant
into it the weights of the old one, we assign IDs to the new architecture such that the layers,
input, and output units that didn't change have the same ID as before, while any additional layer,
input or output unit will have a new ID. Layers of which we change size will not have a new ID as we
want to keep their weights, and it is easy to automatically detect this change.

### Transplant
We implemented the transplant for Dense and Convolutional layers.
- For Dense layers, a change (addition,removal or permutation) in input features’ IDs is
  translated into the columns of the weight matrix. Similarly, a change of output IDs is translated
  into the rows of weight and biases matrices. When adding new inputs or outputs, weights are left
  untouched from the untrained initialization. 
- For Convolutional layers, we use the same technique but inputs’ IDs refer to 
  input channels and output IDs to the layer filters. A change (addition, removal or
  permutation) between the trained layer and new layer of input IDs is applied into each
  filter’s channel. A change of output ids is translated into the same change in the filters
  of the layer. This is done to maintain the operations that each input receives through
  the network.

### Initialize the new layers
The goal is that the new architecture before starting the training behaves as close as possible to the
old one. Dense layers matrices are initialized as close as possible to identities, with zero biases.
For Convolutional layers, we set the filters to be all zero but a one in top-right positions for all
channels. This ensures the input and output are similar enough so the weights previously learned in
posterior dense layers are not compromised.


## Experiments
We performed a set of experiments to test the weight transplant performances. All the experiments have
the following structure:

1. Train a network to solve the problem
2. Modify the network architecture
3. Transplant weights to the new architecture
4. Train the new architecture with the transplanted weights
5. Train the new architecture initialized from scratch
6. Compare results of 1, 4 and 5.

### Increasing the outputs in MNIST classification
First, we train a model to classify only MNIST images that contain numbers from 0 to 3, therefore the
Convolutional Neural Network has 4 outputs. Then, we transplant the weights into a new network that
has 10 outputs, and we train it on the full MNIST dataset.

{% include figure.html url="/_experiments/nn_surgery/mnist.png" %}
