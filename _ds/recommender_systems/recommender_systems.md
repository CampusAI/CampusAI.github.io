---
layout: article
title: "Recommender Systems"
permalink: /ds/recommender_systems
content-origin: mmds.org, towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada
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

Recommender system theory answers the following question:
Given a set of users, items, and their past interactions, how can we recommend the best item for each user?

These past interactions are usually stored in a matrix called the **user-item interaction matrix**.
It can contain ratings of movies, likes/dislikes of videos, times listened to a song...

{% include figure.html url="/_ds/recommender_systems/user-item-matrix.png" description="Each row contains a different user rating of a movie (cols). (Image from buomsoo-kim.github.io/)"%}

{% include annotation.html %}
In our day-to-day life, we are constantly interacting with recommender systems: **YouTube** recommends us new videos, **Amazon** shows us items of interest, **Spotify** proposes playlists...
{% include end-row.html %}
{% include start-row.html %}

## Collaborative filtering methods

Collaborative filtering methods are solely based on the interaction between users and items.
Thus (unlike **content-based methods**) they do not process features of the items.

**Pros/Cons**:
- <span style="color:green">Require no information about users or items.</span>
- <span style="color:green">They improve over time: more interactions $$\rightarrow$$ better recommendations.</span>
- <span style="color:red">**Cold-start** problem: Cannot make predictions to new users (no past interactions available).</span>

Cold-start problem can be addressed in multiple ways:
- **Random strategy**: Recommending random items to new users (or new items to random users).
- **Maximum expectation strategy**: Recommending most popular items to new users.
- **Exploration strategy**: Recommending a set of various items to new users (or a new item to a set of various users).
- Not using a collaborative method for new-coming users.

### Memory-based

Memory-based collaborative filtering usually uses **kNN** (k-nearest neighbor) search from the information on the user-item interaction matrix.
<!-- We represent each user's past interactions as a large sparse vector. -->
<!-- Then, we can recommend popular items amongst the k-closest points in our users dataset. -->

**Pros/Cons**: Same as k-NN algorithm in general:
- <span style="color:green">Low bias.</span>
- <span style="color:red">High variance.</span>
- <span style="color:red">Bad **scalability**: kNN search can be slow if dealing with millions of users and items (need to use approximate kNN methods).</span>
- <span style="color:red">No model $$\rightarrow$$ **lack of regularization**: It is hard to avoid recommending only either popular items or items extremly similar to the ones already consumed.</span>

#### User-User

{% include end-row.html %}
{% include start-row.html %}
Imagine we want to recommend something to a given user $$x$$.
These method would find the most similar users and recommend something these users enjoyed that hasn't been consumed by $$x$$.
This similarity is computed using some distance metric between each user's row in the user-item interaction matrix, accounting for the un-interacted items.

{% include annotation.html %}
Similarity between users is often evaluated using the **centered cosine similarity** (aka [Pearson's coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient))
{% include end-row.html %}
{% include start-row.html %}

We can get an estimation of the rating a user $$x$$ will give to item $$i$$ as:

\begin{equation}
r_{x,i} = \frac{\sum_{y \in \text{SIM}(x)} s_{xy} r_{yi}}{\sum_{y \in \text{SIM}(x)} s_{xy}}
\end{equation}

{% include annotation.html %}
Where:
- $$\text{SIM}(x)$$ represents the subset of users which are similar to $$x$$.
- $$s_{xy}$$ represents the similarity between user $$x$$ and user $$y$$.
{% include end-row.html %}
{% include start-row.html %}

#### Item-Item

Item-item methods use the same idea but from the perspective of items instead of users.
When recommending something, instead of looking at similar users, we can look at similar items to those already liked by the user and recommend these.

{% include end-row.html %}
{% include start-row.html %}

The estimated rating would then be:

\begin{equation}
r_{x,i} = \frac{\sum_{j \in \text{SIM}(i)} s_{ij} r_{xj}}{\sum_{j \in \text{SIM}(i)} s_{xj}}
\end{equation}

{% include annotation.html %}
Where:
- $$\text{SIM}(i)$$ represents the subset of items similar to item $$i$$.
- $$s_{ij}$$ represents the similarity between item $$i$$ and item $$j$$.
{% include end-row.html %}
{% include start-row.html %}

Comparison with user-user:
- <span style="color:green">Item-item **works better** than user-user, as human's complexity tends to be higher than item's one.</span>
- <span style="color:green">Item-item has **less variance**: A lot of users have interacted with an item but each user interacts with few items, so item similarities are less sensitive.</span>
- <span style="color:red">Item-item has **higher bias**: Item similarity is assessed from very different users, so the method is **less personalized**.</span>
 
### Model-based

**Pros/Cons**:
- <span style="color:green">Smaller variance than memory-based methods (less dependent on data).</span>
- <span style="color:red">Higher bias than memory-based methods (assumptions are made when choosing a model).</span>

#### Matrix factorization

{% include end-row.html %}
{% include start-row.html %}
The idea is to factorize the huge (and very sparse) user-item interaction matrix $$M \in \mathbb{R}^{n \times m}$$ into two small and dense matrices:
- The **user-factor matrix** $$Q \in \mathbb{R}^{n\times l}$$: Each row contains the projection of a user into the latent space.
- The **factor-item matrix** $$P \in \mathbb{R}^{m\times l}$$: Each row contains the projection of an item into the latent space.

\begin{equation}
M = Q P^T
\end{equation}

{% include annotation.html %}
In essence, we learn a (linear) model to represent both users and items in a low-dimensional space (of $$l$$ dimensions).
The low-dim features of a user will match those of the movies it enjoys. 
{% include end-row.html %}
{% include start-row.html %}

A possible factorization would be using SVD (check out our [dimensionality reduction post](/ds/dim_reduction_basics)).
However SVD is very **expensive** (considering how big the user-item interaction matrix is) and does not account for all the **missing information**.
Alternatively, we can see it as an optimization problem, where we look for tha matrices which give the best reconstruction:

{% include end-row.html %}
{% include start-row.html %}

\begin{equation}
P, Q = \arg \min_{P, Q} \sum_{(i, j) \in \text{RATINGS}} \left[ P_i Q_j^T - M_{ij} \right]^2 + \underbrace{\lambda_p \Vert P_i \Vert^2 + \lambda_q \Vert Q_j \Vert^2}_{regularization} 
\end{equation}


{% include annotation.html %}
The **regularization** ensures we assign $$0$$ to the unknown interactions.
{% include end-row.html %}
{% include start-row.html %}

We can then find $$P, Q$$ running gradient descent on that loss.
With the advantage that we can train with batches of users and items independently.

Once we have the matrices factorized we can either:
- Predict user $$i$$ rating on item $$j$$ by doing the dot product: $$r_{ij} \simeq P_i Q_j^T$$.
- Run memory-based algorithms (user-user, item-item) on this much smaller and dense vectors.

More generally, we can estimate the rating given by users by fancier operation than dot product such as an ANN $$f_\phi$$:

\begin{equation}
P, Q, \phi = \arg \min_{P, Q, \phi} \sum_{(i, j) \in \text{RATINGS}} \left[ f_\phi (P_i, Q_j) - M_{ij} \right]^2 + \underbrace{\lambda_p \Vert P_i \Vert^2 + \lambda_q \Vert Q_j \Vert^2}_{regularization} 
\end{equation}

## Content-based methods

{% include end-row.html %}
{% include start-row.html %}

Content-based methods process **features** of the items and/or users to find the best recommendations.
In essence, they try to build a model that explains the observed user-item interactions.
Once trained, we can run inference by inputing user features.
The model will output a set of matching items.

**Pros/Cons**:
- <span style="color:green">Suffers far less from **cold-start** problems.</span>
- <span style="color:green">Smaller variance than collaborative methods.</span>
- <span style="color:red">Higher bias than collaborative methods (assumptions are made when choosing features and model).</span>
- <span style="color:red">Harder interpretation.</span>


{% include annotation.html %}
In a song recommender system:
- User features can be: age, sex...
- Item features can be: group, genre, year...
{% include end-row.html %}
{% include start-row.html %}

**Types**:

{% include end-row.html %}
{% include start-row.html %}
- **User-centered**: For each user we train a model using item features. Same as user-user methods, it is much **more personalized** as it is only trained on items the user has interacted with. However, it is **less robust** as each user hasn't interacted with a lot of items.
- **Item-centered**: Train a model for each item which once inptued user features, outputs an estimated rating. Same as item-item methods, it is **more robust** (less variance) but **less personalized** (more bias), as some users with similar features can have dissimilar taste.
- **Combinations**: We can also consider models that process both *user features* and *item features* and guess some affinity.

If looking for simple models we can use **Naive Bayes** or **Logistic regression** for classification scenarios and **linear regression** for regression tasks.
Or we can learn more complex models using ANNs.

{% include annotation.html %}
Item-centered answers the question: *What is the probability of each user to like this item?*, while user-centered methods answer the question: *What is the probability of each item being liked by this user?*.

It is also interesting to note that usually asking for information to new users is much harder than obtaining item information.
Thus, item features tend to be much better than user features.
{% include end-row.html %}
{% include start-row.html %}

## Evaluating Recommender Systems
How do you know if your model is any good?

### Metric-based evaluation

- Split the dataset in train/test subsets and compute (for instance) Categorical Cross-Entropy (in classification tasks) or MSE (in regression tasks) on the training set.
- If the result can be binarized (like/dislike), it is also interesting to analyze the confusion matrix metrics associated with our prediction of the test set: Precision, Accuracy, AUC...
- In the case of recommender systems which do not output a value, but just a ranked list of items (such as collaborative memory-based ones), we can still check if any of the recommendations is present in a test set.

### Human-based evaluation

Another interesting metric to consider is **serendipity**: the diversity of the recommendations (how close are all recommendations between themselves).
- Low serendipity $$\rightarrow$$ all recommended items are very similar, which means our recommender system does not bring enough diversity (aka it creates a **information confinement area**)  
- High serendipity $$\rightarrow$$ recommended items are very different, meaning that the recommender system does not take the user enough into consideration.

Have you ever noticed that Netflix proposes new recommendations explaining why it things it is a good match?
It has been shown that humans loose faith in the recommender system if they do not understand where the recommendations come from.
Thus we also desire models with high **explainability**.

Finally, when introducing changes into the model it is often tested using A/B testing techniques to compare its performance.

{% include end-row.html %}
