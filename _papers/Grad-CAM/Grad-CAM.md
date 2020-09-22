---
layout: paper
title: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
category: Explainability
permalink: /papers/Grad-CAM
paper-author: Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra
post-author: Oleguer Canal
paper-year: 2019
paper-link: https://arxiv.org/abs/1610.02391
---
<!--
Disclaimer and authorship:
This article is provided for free only for your personal informational and entertainment purposes. No commercial use of it is allowed.

Please note there might be mistakes. We would be grateful to receive (constructive) criticism if you spot any. You can reach us at: ai.campus.ai@gmail.com or directly open an issue on our github repo: https://github.com/CampusAI/CampusAI.github.io

If considering to use the text please cite the original author/s of the lecture/paper.
Furthermore, please acknowledge our work by adding a link to our website: https://campusai.github.io/ and citing our names: Oleguer Canal and Federico Taschin.
-->

## IDEA

ANNs lack of decomposability into independent components makes it very challenging to understand their "reasoning".
Visual explanations (VE) can help with this issue in every state of AI development.
Compared to humans, when AI is:
1. **Weaker**: VE is useful to **detect failure models**.
2. **On pair**: VE is useful to **establish trust** to their users. (e.g. Image recognition with enough data)
3. **Stronger**: VE is useful to **explain** to humans, known as machine teaching. (e.g. Chess or GO)

This paper develops on this challenge with 2 contributions:
1. Generalizes the [Class Activation Mapping (CAM)](https://arxiv.org/abs/1512.04150) method for detecting attention areas.
2. It combines the previous algorithm with [Guided BackPropagation](https://arxiv.org/abs/1412.6806) to detect the key features of the relevant area.


Consider an image classification task modelled by a CNN.
Given an input image and a label, [CAM](https://arxiv.org/abs/1512.04150) provides a heatmap over the image of the area which is more relevant for that particular label.
Nevertheless its applicability is quite limited.
It only supports fully CNN models, i.e. models which do not have any Dense layer
This paper presents a more general approach which works on any architecture whose first model is a CNN.


