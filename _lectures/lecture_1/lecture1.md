---
layout: lecture
title: "Lecture 1: Introduction"
permalink: /lectures/lecture1
lecture-author: Sergey Levine
lecture-date: 2019
slides-link: http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-1.pdf
video-link: https://www.youtube.com/watch?v=SinprXg2hUA&list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A&index=2&t=0s
---

<!-- # Reinforcement Learning vs Supervised Learning -->

- **Deep Learning (DL):** Useful for recognition of unstructured data.
<!-- (*raw* data not designed for a machine to work well on) -->
- **Reinforcement Learning (RL):** Useful for decision-making problems:  An **agent** interacts with an **environment**. It perceives **states** and takes **actions** according to some **policy** to maximize a sense of **reward**.

{% include figure.html url="/_lectures/lecture_1/idea.jpg" description="RL problem structure" %}

**OBS:** Most problems we perceive as AI problems can be framed as RL problems. (e.g. Image classification can be seen as a decision problem with +1 reward when correctly classified).

**OBS:** Deep models are what make RL solve complex tasks end-to-end. They allow the direct mapping between states and actions.

<!-- ## RL Examples:

Robot:
- **Actions:** Joint positions
- **Observations:** Camera images
- **Rewards:** Task success measure

Inventory management:
- **Actions:** What to purchase
- **Observations:** Inventory levels
- **Rewards:** Profit -->

## Ways to learn:

- From expert demonstrations:
    - **Behavioral Cloning**: Copying observed behavior. [Lecture 2](/lectures/lecture2)
    - **Inverse RL**: Inferring rewards from observed behavior.


- From environment observations:
    - **Model Learning**: Learn a model of the environment to then find an optimal policy.
    <!-- "Learning to predict" -->
    - **Unsupervised Learning**: Observe the environment and extract features


- From other tasks:
    - **Transfer Learning**: Share knowledge between different tasks
    - **Meta-learning**: Learning to learn: Use past learned tasks to optimize the learning process.

## How to build intelligent machines:

- **Classic engineering approach**: Split problem into easier sub-problems, solve them independently and wire them together.
    - **Problems:**
        - Sub-problems in intelligent systems can already be too complex.
        - Wiring of non-intentional abstracted blocks may rise issues.

- **Learning-based approach**: Use learning as a base for intelligence.
    - **Motivations:** 
        - While some human behaviors may be innate (walking), others can only be learned (driving a car is clearly not programed into us by evolution) $\Rightarrow$ humans have learning mechanisms powerful enough to perform everything associated with intelligence.
    - **Problems:**
        - Still might be convenient to *hard-code* some bits.
    - **Doubts:**
        - Should we still split the problem into sub-domains and apply different learning algorithms to each one (e.g. one for perception, one for locomotion...) or use a single learning algorithm which acquires the functionality of these subdomains? There is some evidence supporting a single learning method:
            - Resemblance between features extracted by Deep Neural Nets (DNN) and primary cortical receptive fields. [Andrew Saxe et al.](https://papers.nips.cc/paper/4331-unsupervised-learning-models-of-primary-cortical-receptive-fields-and-receptive-field-plasticity)
            - [Re-wiring optical nerve to auditory cortex](http://web.mit.edu/surlab/publications/Newton_Sur04.pdf). This experiment shows how a mammal can re-learn to process its eyes information using its auditory cortex. 
            - With [BrainPort device](https://www.youtube.com/watch?v=xNkw28fz9u0) you can "see through your tongue". It converts images into electric signals perceived by the tongue.


## State of the art:
### What can DRL do well?
- High proficiency in domains with simple and **known rules**:
[AlphaGo](https://www.youtube.com/watch?v=WXuK6gekU1Y), [Atari Games](https://deepmind.com/blog/article/Agent57-Outperforming-the-human-Atari-benchmark)
- Learn **simple skills** with raw sensory inputs (given enough experience) [Sergey Levine Research](https://people.eecs.berkeley.edu/~svlevine/)
- Learn by **imitating** enough human expert behavior

### What are main DRL challenges?
- DRL methods are very **slow**: They require a lot of experience samples before they work well.
- Humans re-use a lot of knowledge but **transfer** learning in DRL is still an open problem.
- How should the **reward** function be? What is the role of **predictions**?
