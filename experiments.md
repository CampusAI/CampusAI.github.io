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

{% include card.html
title="Self-learned vehicle control using PPO"
img="/_experiments/autonomous_driving/icon.gif"
url="/experiments/autonomous_driving"
brief="This work tackles the completion of an obstacle maze by a self-driving vehicle in a realistic physics environment.;
We tested the adaptability of our algorithm by training both a car and a drone.;
"
type="description" %}

