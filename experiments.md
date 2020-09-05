---
layout: page
title: Experiments
permalink: /experiments
---
<!--
To add an experiment one must add a line with the following code:

include media_card.html title="" brief="" img="" url="" type=""

title:  The title of the lecture 
brief:  A string of ; separated sentences that will be put in a bullet list
img:    An image that represents the lecture
url:    The url of the lecture post
type:   The type of the card. Here we use "description".
-->
This page contains some of the RL-related experiments we did.
We mainly test algorithms on new environments and re-implement papers code.

Stay tunned for new experiments such as a RL-trained [**RocketLeague**](https://en.wikipedia.org/wiki/Rocket_League) bot and a **Terrarium** where animals were trained to survive while following a simple food chain.

{% include card.html
title="Self-learned vehicle control using PPO"
img="/_experiments/autonomous_driving/icon.gif"
url="/experiments/autonomous_driving"
brief="This work tackles the completion of an obstacle maze by a self-driving vehicle in a realistic physics environment.
We tested the adaptability of our algorithm by learning both the controls of a car and a drone.
The agents are trained using the PPO (policy gradient) algorithm and using curriculum learning for faster training.;
This project was part of a contest and we achieved the faster maze completion times against heuristic approaches.
"
type="description" %}

{% include card.html
title="Neural Network Surgery in Deep Reinforcement Learning"
img="/_experiments/nn_surgery/lunar_lander.gif"
url="/experiments/nn_surgery"
brief="In this work we experiment on weight transplantation after slight ANN modifications.
Networks can be modified by increasing or reducing the number of input/outputs as well as the number of hidden layers and units.
We show that when modifying the network structure, weight transplant achieves faster and better results than training from scratch."
type="description" %}