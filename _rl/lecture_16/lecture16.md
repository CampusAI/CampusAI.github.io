---
layout: lecture
title: "Transfer and multi-task RL"
permalink: /lectures/lecture16
lecture-author: Sergey Levine
lecture-date: 2019
post-author: Oleguer Canal
slides-link: http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-16.pdf
video-link: https://www.youtube.com/watch?v=eeww07Jxncw&list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A&index=15
---
<!--
Disclaimer and authorship:
This article is provided for free only for your personal informational and entertainment purposes. No commercial use of it is allowed.

Please note there might be mistakes. We would be grateful to receive (constructive) criticism if you spot any. You can reach us at: ai.campus.ai@gmail.com or directly open an issue on our github repo: https://github.com/CampusAI/CampusAI.github.io

If considering to use the text please cite the original author/s of the lecture/paper.
Furthermore, please acknowledge our work by adding a link to our website: https://campusai.github.io/ and citing our names: Oleguer Canal and Federico Taschin.
-->
{% include start-row.html %}

**NOTE**: Transfer Learning (TL) & Multi-task Learning (MTL) are still an open problem in DRL, this lecture is more a survey on latest research rather than some well-established methods.

**TL Terminology**:
- **Task**: In the RL context, task is the same as MDP.
- **Source domain**: Initial task where the agent has been trained.
- **Target domain**: Task which we want to solve. (We aim to transfer knowledge from source to target domain).
- **n-shot**: Number of attempts in the target domain before the agent is able to solve it. (often used as 0-shot, 1-shot or few-shot)

## TL in RL
Prior understanding of problem structure can help us solve complex tasks quickly and have better performance.

Where and how can we store knowledge?
- **Q-function**: Remember which state-action pairs are good
- **Policy**: Remember which actions are useful in a given state (and which are never useful)
- **Models**: Remember physics laws, so you do not have to learn them again
- **Features/hidden states**: Re-using ANN feature extractors when re-training an agent (one of the most common uses)

In this [Loss is its own Reward: Self-Supervision for Reinforcement Learning](https://arxiv.org/abs/1612.07307) they show its effectiveness in vision-based tasks (although only in the same kind of task):

{% include end-row.html %}
{% include start-row.html %}

{% include figure.html url="/_rl/lecture_16/feature_extraction_keeping.png" description="Evolutions when training a policy from scratch (black) vs using pre-trained weights for all except last layer (red). As expected using pre-trained feature extractors significantly speeds the learning process. Image by OpenAI."%}

{% include annotation.html %}
This example shows the two jobs the model is doing:
1. Extract features from observations.
2. Extract value from these features.

By inputing extracted features, the model learns much quickly as it only needs to focus on task 2.
{% include end-row.html %}
{% include start-row.html %}

## Forward transfer

**In short:** Train on one task, transfer to a new one

{% include end-row.html %}
{% include start-row.html %}
In **supervised learning**, we usually perform TL as follows:
1. Train a large network in a large dataset (e.g. train a big resnet on imagenet)
2. Substitute last layers of the networks for randomly-initialized ones.
3. Depending on the size of our dataset:
   1. If small: train only updating the weights of the last new layers
   2. If large: train all weights 

{% include annotation.html %}
This methodology is very effective in vision tasks.
{% include end-row.html %}
{% include start-row.html %}

**Problems**:
- <span style="color:red">**Domain shift**</span>: Representations learned in the source domain might not work on the target domain. Usually, we take the **invariance assumption**: everything that is **different** from both domains is **irrelevant**. While not usually true, if it was true, there exist some things we can do to deal with domain shift. For instance, we can train using **domain confusion** (aka domain-adversarial NN): Instead of updating the weights just from the main task, we can also use the reversed gradient of a classification tasks which learns to tell apart both domains from a latent representation of the input.

But, how does that idea translate into RL?


### Naive approach

{% include end-row.html %}
{% include start-row.html %}

Train a policy on the source domain and test it on the target domain (hoping for the best).
Usually does not work.

{% include annotation.html %}
More on [End-to-End Training of Deep Visuomotor Policies](https://arxiv.org/abs/1504.00702)
{% include end-row.html %}
{% include start-row.html %}


### Finetuning

Train on a task, re-train on a new one using the first feature-extraction layers of the ANN (essentially the same explained before but changing tasks).
It is specially useful for vision-based policies.
One can even train the feature-extraction layers in a supervised learning setup with some famous dataset.

**Problems**: 
- <span style="color:red">Too **specialized** policies.</span> If the domains are too distinct weight transfer can worsen things since policies tend to become very **specialized**.
- <span style="color:red">Too **deterministic** policies.</span> Some methods (e.g. policy gradient) favour a lot good actions while lowering the probability of the bad ones: when moved to a new domain they will not explore enough. This gets improved by using **maximum-entropy policies**, which try to maximize rewards while acting as randomly as possible.

Something that also improves performance is to pre-train for robustness, for instance: try to learn solving tasks in **all possible ways**:

{% include figure.html url="/_rl/lecture_16/robustness.png" description="Learning multiple ways to solve first (right) maze will help should it be changed as in the second (left) image. Obtaining thus, a more robust transfer."%}

{% include end-row.html %}
{% include start-row.html %}
When finetuning we usually do not want to maximize entropy as we usually want to squeeze the best performance out of our agent.
Usually in the [Maximum Entropy Framework](/papers/Soft-Actor-Critic) we have a temperature parameter to tune the amount of entropy desired.
If the output of our ANN is a multivariate Gaussian, we can reduce its variance. Though it has not been trained that way it usually works.

{% include annotation.html %}
More on [Reinforcement Learning with Deep Energy-Based Policies](https://arxiv.org/abs/1702.08165) and in [DARLA: Improving Zero-Shot Transfer in Reinforcement Learning](https://arxiv.org/abs/1707.08475).
{% include end-row.html %}
{% include start-row.html %}

### Domain manipulation

If we have a very difficult target domain we can **design our source domain** to improve its transfer capabilities.
Same idea remains: The more diversity we see, the better will the transfer be.

For instance, if training on some physical system, one can use a source domain with randomized variables (e.g. weight and sizes) to make a more robust transfer: [EPOpt: Learning Robust Neural Network Policies Using Model Ensembles](https://arxiv.org/abs/1610.01283)

{% include figure.html url="/_rl/lecture_16/var_randomization.png" description="Variable randomization example in hopper environment."%}

Other ideas, such as the ones presented in: [Preparing for the Unknown: Learning a Universal Policy with Online System Identification](https://arxiv.org/abs/1702.02453) rely on learning some physical parameters of the environment in a supervised manner in source domains, which are also feeded into the policy.
Later, on the target domain they are estimated.
Results show that this approach achieves almost the same performance as directly feeding the real parameters to the policy. 

Some [ideas](https://arxiv.org/abs/1611.04201) explore visual randomization of the environment. For instance, in simulation, change walls textures to have a more robust policy.

More on simulation to real RL in this [paper](https://arxiv.org/abs/1710.06537).

### Domain adaptation

So far we talked about pure 0-shot transfer:
Learn in source domains so we can succeed in **unknown** target domains, we thus needed to be as robust as possible. Nevertheless, we can usually sample some states (e.g. images) from our target domain and do something smarter than pure randomization.

[Adapting Deep Visuomotor Representations with Weak Pairwise Constraints](https://arxiv.org/abs/1511.07111) applies the **domain confusion** (aka adversarial domain adaptation) idea we previously presented to RL.
They synthetically create simplified states in the source domain to train on.
Later, they force the model to represent both synthetic and real states by the same kind of features.
To do so in a vision task: you impose some loss on the higher level of the CNN which forces the extracted features of images from different domains to look similar to each other.

{% include figure.html url="/_rl/lecture_16/sim2real.png" description="Example of a synthetic simplified state and its corresponding real one in a vision task."%}

A way to do this is by automatically pairing simulated images with real images and regularize the features together to get similar values.

**Problem**: <span style="color:red">Usually we do not have this kind of knowledge.</span> We just have samples of both synthetic and real state extracted with some policy (e.g. random). They are not from the same state but both come from the same sampling distribution.

**Solution**: <span style="color:green">Use an **unpaired distribution alignment loss** between features</span> (e.g. use some kind of GAN loss). You can achieve that by putting a small discriminator on the features of your network as depicted:

{% include figure.html url="/_rl/lecture_16/domain_adaptation.png" description="Adversarial domain adaptation network adaptation."%}

This other [approach](https://arxiv.org/abs/1709.07857) explicitly uses a GAN to convert the simplified state images into realistic ones in a pixel to pixel fashion.
The generator gets inputed the simplified states and makes them look as similar as possible to the source domain state images.
The discriminator then tries to take apart the real from the generated images.
Creating thus a competition between generator and discriminator which forces the generated images to get similar to the real ones.


## Multi-task transfer

**In short:** Train on many tasks, transfer to a new one

So far, we need to design (simulation) diverse very similar tasks to achieve a good transfer.
But humans do not work this way!
We transfer knowledge from multiple **different** tasks.
This is very hard with our current algorithms, we can get what is known as **negative transfer**: using pre-trained weights actually harms the learning of the target task.

### Transfer in Model-based RL

Even if all past tasks are different they might have things in common, for instance: the **same laws of physics** (e.g. same robot doing different chores / some car driving to different destinations / different tasks in same open-ended video game).

{% include end-row.html %}
{% include start-row.html %}
In model-based RL we can train an environment model on past tasks and use it to solve new tasks. Sometimes we'll need to fine-tune the model for new tasks (easier than fine-tunning the policy).

{% include annotation.html %}
More on [One-Shot Learning of Manipulation Skills with Online Dynamics Adaptation and Neural Network Priors](https://arxiv.org/abs/1509.06841).
{% include end-row.html %}
{% include start-row.html %}
### Learn a multi-task policy

The idea is to learn a policy which solves multiple tasks with the hopes that:
- It will be easier to fine-tune to new tasks.
- Learning multiple tasks simu

#### Train single policy in a joint MDP

The most basic approach would be to construct a single MDP which describes all tasks.
This MDP is composed by multiple smaller MDPs (one for each task).
Then you train your policy on the big MDP, where the first state tells you in which task you are.

{% include figure.html url="/_rl/lecture_16/multiple_mdps.png" description="Joint MDP approach."%}

**Example**: You can train an agent to play multiple Atari games (each one having its own MDP), then in the first state you just sample one game to play.

**Problems**:
 <!-- Learning a single policy to play all these games can be very challenging! -->
- <span style="color:red">Gradient interference:</span> Becoming better at one task can make you worse at another.
- <span style="color:red">Winner-take-all problem:</span> If a task starts getting good, the algorithm may start to prioritize that task to increase average expected reward, harming the performance of the others. (Snowball effect)

Therefore, in practice attempting this kind of multi-task RL is very challenging.

#### Train multiple policies in different MDPs and join them

As explained in [Model-based policy learning lecture](/lectures/lecture12) we can use **distillation for multi-task transfer**.
Essentially, you train a policy for each of the domains you have and use supervised learning to create a single policy from the multiple learned ones.

{% include figure.html url="/_rl/lecture_16/distillation.png" description="Distillation idea scheme."%}

{% include end-row.html %}
{% include start-row.html %}
We want to find the parameters which make the joint policy distribution be as close as possible to the original ones.\
Thus, we can run MLE on the following cross-entropy loss:

\begin{equation}
\mathcal{L} = \sum_a \pi_i (a \mid s) \log \pi_{AMN} (a \mid s)
\end{equation}

{% include annotation.html %}
The idea is similar to **Behavioral Cloning** ([BC lecture](lectures/lecture2)), but copying learned policies instead of human demonstrations.
{% include end-row.html %}
{% include start-row.html %}

{% include end-row.html %}
{% include start-row.html %}
Results show that depending on the game the transfer is beneficial or not.
In particular, it is beneficial among those games which share similar traits and counterproductive in those that stand out for being different than the others.

{% include figure.html url="/_rl/lecture_16/distillation_transfer.png" description="Distillation transfer results (blue) vs standard methods (red, yellow, purple)."%}

{% include annotation.html %}
More on [Actor-Mimic: Deep Multitask and Transfer Reinforcement Learning](https://arxiv.org/abs/1511.06342)
{% include end-row.html %}
{% include start-row.html %}

### Contextual policies

Sometimes instead of solving tasks in different environments, we need our agent to solve different tasks in the same environment.
In this case we need to indicate which goal we want our agent to achieve.
We can do that by adding some indication in the policy and converting it into a **contextual policy**:

\begin{equation}
\pi_\theta (a \mid s) \rightarrow \pi_\theta (a \mid s, w)
\end{equation}

Formally, this simply defines and augmented tate space $$\hat s = [ s, w ]$$ where $$\hat S = S \times \Omega$$, where we simply append the context.

### Multi-objective RL within the same environment

It is common that we want to solve different tasks within the same environment.
More formally, we have constant dynamics $$p(s_{t+1} | s_t, a_t)$$ but different reward functions (one for each task).
For instance, we have a kitchen robot that has to cook multiple recipes: while the setup is the same, each recipe has a different reward function associated with it.

In this case, what is the best object to transfer?

#### Model
If doing model-based-RL, it, should be the easiest thing to transfer, as (in principle) it is independent of the reward.
While this works, it still presents some problems if attempting 0-shot transfer:
- <span style="color:red">Exploration limitations</span>: Maybe in the source domain we only visited the subset of states which were relevant for the source task, the target task might need to use other states which the model hasn't been trained on.

#### Value Function $Q$
{% include end-row.html %}
{% include start-row.html %}
Might be tricky since it entangles **environment dynamics**, **policy** and **reward**.
Nevertheless, it turns out that usually these things are coupled linearly, so it is not that hard to disentangle.
Introducing **successor features**:

{% include annotation.html %}
These ideas are presented in [Successor Features for Transfer in Reinforcement Learning](https://arxiv.org/abs/1606.05312)
{% include end-row.html %}
{% include start-row.html %}

Lets remember the Bellman equation:

\begin{equation}
Q^\pi (s, a) = r (s, a) + \gamma E_{s^\prime \sim p(s^\prime \mid s, a), a^\prime \sim \pi (s^\prime)} \left[ Q(s^\prime, a^\prime) \right]
\end{equation}

Lets consider a vectorization of the value function: $$q = \text{vec} \left( Q^\pi (s, a) \right) \in \mathbb{R}^{\vert S \vert \cdot \vert A \vert}$$.
We then can linearly approximate the Bellman equation as:

{% include end-row.html %}
{% include start-row.html %}

\begin{equation}
q = \text{vec} (r) + \gamma P^\pi q
\end{equation}

{% include annotation.html %}
Notice that if the state or action spaces are continuous this still holds for infinite-dimensional vectors.
{% include end-row.html %}
{% include start-row.html %}

We call $$P^\pi$$ the transition probability matrix according to policy $$\pi$$.
Then we have that:

\begin{equation}
q = (I - \gamma P^\pi )^{-1} \cdot \text{vec} (r)
\end{equation}

Imagine the different tasks we want to solve have reward functions which are a linear combination of $$N$$ different features: $$\text{vec} r(s, a) = \phi w$$ ($$\phi$$ is a matrix that encodes this relationship).
We can then define a matrix $$\psi := (I - \gamma P^\pi )^{-1} \cdot \phi$$.
Effectively linearly disentangling the dynamics from the reward:

\begin{equation}
Q^\pi = \psi w
\end{equation}

We say that the i-th column of $$psi$$ is a **"successor feature"** for the i-th column of $$\phi$$.
So for any linear combination $$\phi$$, we can get a Q-function by applying the same combination on $$\psi$$.

Notice this works on a value policy $$Q^\pi$$ but it doesn't on the optimal value function $$Q^\star$$.
Remember that:

\begin{equation}
Q^\star (s, a) = r (s, a) + \gamma E_{s^\prime \sim p(s^\prime \mid s, a)} \left[ \max_{a^\prime} Q(s^\prime, a^\prime) \right]
\end{equation}

The $$\max$$ function makes the expression to be NOT linear!

How do we learn this in practice?

<blockquote markdown="1">
**Simplest use**: Evaluation
1. Get a small amount of data of the new MDP: $$\{ (s_i, a_i, r_i, s_i^\prime)\}_i$$
2. Fit $$w$$ such that $$\phi (s_i, a_i) w \simeq r_i$$ (e.g. with linear regression)
3. Initialize $$Q^\pi (s, a) = \psi (s, a) w$$
4. Finetune $$\pi$$ and $$Q^\pi$$ with any RL method
</blockquote>

{% include end-row.html %}
{% include start-row.html %}
<blockquote markdown="1">
**More sophisticated use**:
0. Train multiple $$\psi^{\pi_i}$$ functions for different $$\pi_i$$
1. Get a small amount of data of the new MDP: $$\{ (s_i, a_i, r_i, s_i^\prime)\}_i$$
2. Fit $$w$$ such that $$\phi (s_i, a_i) w \simeq r_i$$ (e.g. with linear regression)
3. Choose initial policy $$\pi (s) = \arg \max_a \max_i \psi^{\pi_i} (s, a) w$$
</blockquote>

{% include annotation.html %}
Notice that if the state or action spaces are continuous this still holds for infinite-dimensional vectors.
{% include end-row.html %}
{% include start-row.html %}

#### Policy
While doable with contextual policies, it might be tricky otherwise, since it contains the "least" dynamics information.

### Architectures for multi-task transfer

So far we only aimed to model a policy for multiple tasks with a single ANN.
But what if the tasks are fundamentally different? (e.g. they have different inputs)
Can we design architectures with **reusable components**? (to then re-assemble them for different goals)

Introducing **modular networks in deep learning** ([paper](https://arxiv.org/abs/1511.02799)).
They present a (quite engineered) algorithm to answer questions of a given image.
They combine modules for finding text entities in images and modules to extract purpose of the question to provide an answer.

This [work](https://arxiv.org/abs/1609.07088) brings this idea to RL.
In a setup with $$n$$ robots working on $$m$$ tasks instead of having a single ANN policy which learns to deal with any case, we can de-couple the robot control from the task goal and learn each one in a modular fashion:

{% include figure.html url="/_rl/lecture_16/modular_ann.png" description="Modular ANN training approach. Different robots learn different tasks in a modular setup. Notice that you only need 2 different modules (robot and task) to learn 4 situations."%}

The performance of this approach depends on how many different values you have of this factors of variation and the information capacity between the modules is not too large (otherwise you need to use some kind of regularization).

{% include end-row.html %}