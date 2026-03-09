# Predictive Alerting for Cloud Metrics

## Problem Formulation
The goal is to predict cloud infrastructure incidents before they happen to reduce downtime. I framed this as a binary classification problem over time series data. Given a lookback window **W** of 15 time steps (minutes), the model predicts if an incident will occur within the prediction horizon **H** of the next 5 time steps. I generated a synthetic dataset simulating CPU usage with embedded seasonality, random noise, and injected spikes that lead directly to failure states.

## Modeling Choices
I engineered rolling statistical features (mean, standard deviation, max) and the first order derivative (gradient) over the window **W** to capture the trajectory of the metric, not just its current state. I chose a Random Forest Classifier using Scikit Learn for a few key reasons:

* It handles tabular sliding window features extremely well without requiring massive amounts of data like deep learning models.
* It captures nonlinear relationships (e.g. CPU is only dangerous if both usage and standard deviation spike together).
* I used `class_weight='balanced'` because incidents are rare, and standard models tend to heavily bias towards the majority "healthy" class.

## Evaluation Setup and Alert Thresholds
In cloud alerting, accuracy is a terrible metric. If a system is healthy 99% of the time, a model that never alerts is 99% accurate but completely useless. 

Therefore, I evaluated the model using Precision, Recall, and the Precision Recall AUC (PR-AUC). 

* **Recall** measures how many actual incidents we successfully caught before they happened.
* **Precision** measures how many of our alerts were actually real. 

In a real world scenario, I would tune the decision threshold probability to heavily favor Precision to prevent "alert fatigue" for the DevOps team. If we flood the Slack channel with false positives, engineers will start ignoring the alerts.

## Results Analysis and Real World Adaptation
The model successfully identifies the pre incident spikes, scoring highly on the PR-AUC curve. Moving this approach to a real alerting system would require a few architectural steps:

1. Deploying the model behind an API endpoint that ingests live Prometheus or Datadog metrics streams.
2. Switching to a more performant streaming feature extraction pipeline (perhaps using Redis to maintain the rolling window **W** state) rather than computing Pandas rolling windows in batch.
3. Implementing a cooldown period logic post alert so the system does not spam the on call engineer for the same ongoing anomaly within the **H** window.
