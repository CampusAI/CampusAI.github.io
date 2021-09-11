---
layout: article
title: "ML in production"
permalink: /ml/ml_in production
content-origin: KTH DD2412
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

Deploying and maintaining ML models is quite a software engineering intensive task (often overlooked in ML courses).
In an attempt to sort some ideas, this post summarizes the main concepts using the [GCP](https://cloud.google.com/) framework.

## Interview example walkthroug

Most interviews include a use-case example of ml in production.
These are the 

### Metrics
What metrics should be used for modelling?
- **Offline metrics**: Used to score the model when building it. Those are obtained on pre-stored data. `offline eval`
- **Online metrics**: Scored that we get from the model once it is in production. `grafana`
- **Non-functional metrics**: Training speed, scalability, modularity, tooling available (for training, debugging, evaluation, deployment)...

### Data

**Training data**:
- Identify **target variable** (usually differentiated between implicit or explicit, depending on user interaction tracked)
- Think **features** to use (feature selection can be done through tree-based estimators, L1 regularization...)
- **Feature engineering**: Train-test split, missing values/outliers, class balance, normalization...
- Consider **biases**, **privacy**, **law** issues...

**Data storage**:
- **Object storage**: Such as models/data dictionaries/sound/images/video: `Amazon S3`, `GCP Cloud Storage`

{% include end-row.html %}
{% include start-row.html %}

- **Structured data**: Relational database where information is stored in tables `GCP BigQuery` (optimized for large amounts of relational structured data)

{% include annotation.html %}
This is what I use to create testing sets, gather info from a relational database (one or more tables) using SQL commands.
{% include end-row.html %}
{% include start-row.html %}

{% include end-row.html %}
{% include start-row.html %}

- **Non-SQL** (non-relational database) **wide-column** (each row can have different cols):  `GCP BigTable` (optimized for large read/write requests)

{% include annotation.html %}
Related data doesn't need to be split in different tables, everything can be in the same one: NoSQL data models allow related data to be nested within a single data structure.
{% include end-row.html %}
{% include start-row.html %}

### Model

1. Think of a baseline "model": Something which does not require ML (s.a. returning most popular item) and all others should outperform
2. Think of simple models easy to implement and fast to train. Think of pros and cons of each approach.

### Serving

Most common points to mention are:
- A/B testing on online metrics
- Decide where to run inference:
  - **User's device**: Reduces latency at the cost of their memory & battery
  - **Our own service**: Privacy concerns.
- How to monitor performance
- Possible biases and misuses of the model.
- Model re-training

Common ML operations are:
{% include end-row.html %}
{% include start-row.html %}
- Store logs in a database like `ElasticSearch` (APM Application Performance Management)
- Logging analytics `Grafana`
- CI/CD `CircleCI`

{% include annotation.html %}
`ElasticSearch` is a text search engine written in Java. Provided a database it implements common search functionalities.
{% include end-row.html %}
{% include start-row.html %}

## Puzzle pieces

### Feature store

{% include end-row.html %}
{% include start-row.html %}

Used to manage datasets and pipelines needed to productionize ML applications: *“It is the interface between models and data”*.
<!-- In Spotify: `Jukebox` is the repo which implements components to rapidly set up a feature store.
It heavily relies on:
- `Luigi` Open-source python orchestration framework
- `Hades` Service to manage data endpoints -->

{% include figure.html url="/_ml/ml_in_production/feature_store.png" description="Feature store overview. Image from https://www.tecton.ai/blog/what-is-a-feature-store/" width="200" zoom="1.0"%}

<!-- {% include annotation.html %}
- To find a dataset containing the needed information in spotify use `Lexikon`.
- To run SQL queries from a python notebook spotify developed `bqt` (big query tool) useful for direct data analysis later on.
{% include end-row.html %}
{% include start-row.html %} -->

{% include end-row.html %}
{% include start-row.html %}
There are 5 main components:

- **Serving**: Provide feature data to models (needs to be consistent across training and serving).
  - In **serving** time usually done through REST API.
  - In **training** time usually through some python SDK.
- **Storage**: Retain offline and online feature data.
  - **Offline storage** is used for training purposes `BigQuery`.
  - **Online storage** are used for low-latency lookup during inference usually using key-value stores like `Cassandra`.
- **Transformation**: Processing of raw data into feature values.  `Apache Spark`, `Tensorflow Transform`, `Scio`
  - **Batch transformation** is done at data at rest.
  - **Streaming transform** is applied to streaming sources.
  - **On-demand transform** are transformation only available at the time of prediction.
- **Monitoring** data correctness through user-defined schemas, data quality for drift and skews...
- **Registry**: Common catalogue used by teams to explore, develop, and publish new definitions. It also contains metadata 

{% include annotation.html %}
- **API** (Application Programming Interface): set of rules that define how applications can communicate. There are different styles:
  - **REST API**: API which conforms design principles of REST (representational state transfer architectural style) such as
uniform interface, client-server decoupling (only thing server knows is the uri sent by client), statelessness (each request needs all information)... It mainly relies on JSON and XML.
  - **gRPC API**: Uses Protocol Buffer by default to serialize payload data.
{% include end-row.html %}
{% include start-row.html %}

### Training

Basic tools:
- **Docker**: Open-source technology for automating deployment of applications as self-sufficient containers.
- **Kubernetes**: Open-source orchestration software that provides an API to control how and where those containers will run. It makes it easier to manage multiple containers across multiple servers.
- **Kubeflow**: Open-source kubernetes-native platform to develop/run/orchestrate/deploy scalable ML workloads.
- **Kubeflow Pipelines**: Kubeflow component to manage end-to-end workloads. It includes:
  - **SDK** to manage the workload from python `$ skf command`
  - **WEB UI** to monitor pipelines accessible through the Kubeflow dashboard
<!-- - **Spotify Kubeflow**: Wrapper around kubeflow to accelerate and ease ML pipelines execution. 
  - Python **SDK**: With already-implemented common components in `spotify_kubeflow.component.common`. Usually `import spotify_kubeflow as skf`
  - **CLI** (Command Line Interface) to execute tasks from the terminal
  - Set of Kubeflow clusters hosted in GCP for job deployment -->


Pipeline concepts:
- **Component**: Self-contained set of code which represents a step in a ML workflow (e.g. data preprocessing, data transformation, model training...).
- **Pipeline**: Includes all the workflow components and how they relate to each other in the form of a graph. Usually there are some common shared parameters (Pipeline Options), in addition each component has its own arguments.
- **Run**: Single execution of a pipeline
- **Experiments**: Group of runs

ML platforms:
- **TensorFlow**: Open-source platform for ML.
- **TensorFlow Extended (TFX)** Opinionated workflow for production of ML pipelines.

{% include end-row.html %}
{% include start-row.html %}

{% include figure.html url="/_ml/ml_in_production/tfx.png" description="TFX pipeline steps and its main components." width="200" zoom="1.0"%}

{% include annotation.html %}
- While the pipeline runs in the **Kubeflow** cluster, different components happen at different GCP products.
For instance, components which use `Apache Beam` run on `GC Dataflow`, training and hyper-parameter tunning happens on `GC AI Platform`.
<!-- - `HadesImporter` can be used to use ouputs from components of different pipelines. -->
{% include end-row.html %}
{% include start-row.html %}

- Model evaluation is done using `TFMA` (TensorFlow Model Analysis). It provides metrics on slices of data (you can ) and deals with large amounts of data using `Apache Beam`.
- Model validation ensures the model meets a minimum of standards to be pushed (aka _"is blessed"_).

Pipelines:
- **Dataset creation pipeline**: Runs at regular intervals before the training pipeline.
- **Training pipeline**: Runs at regular intervals (daliy, weekly...)
- **Batch prediction pipeline**: Can be used for model analysis or to pre-compute outputs.

<!-- 
### Serving

- Features are served through the feature store.
- You also need a backend service that runs your model. In Spotify, this is done through `Salem`: It takes an ML model and provides a backend service where you can send features and get the predictions. Characteristics:
  - The definition of how the model should be deployed (memory, CPU capacity...) is set on a **problem configuration** file.
  - **Model server**: Performs real-time inference and runs on top of `Tensorflow Serving`.
  - `Slots` named container for a model server (a problem can have multiple slots, running different ML models)
  - Has a **model deployer**, which detects any new version of your model and updates it. It will also create `Docker` containers for `Salem API` and `Slots`

### CI/CD -->


<!-- 
## Spotify

- `scio` Scala API for `Apache Beam` and `GC Dataflow` (Similar to `Apache Spark`). -->

{% include end-row.html %}
