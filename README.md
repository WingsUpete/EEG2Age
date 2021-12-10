# EEG2Age
Human Brain Age Prediction based on EEG Signals

<br>

## Introduction

This is a project from a course BME5012 - Brain Intelligence & Machine Intelligence (2021 Fall), in SUSTech. The project aims to study several deep neural network designs to predict the human brain age according to the given EEG brain signals. Currently, the common approach is to train certain model using the chronological age of healthy human as the label. Then, the model can be used to analyze the "brain age" of a patient so as to discover whether there is any functional disorder occurring.

<br>

## Data

xxx

<br>

## Model

### Our Model

xxx

### Baseline

We use a simple **feed-forward neural network** which contains only two linear layers as the baseline model.

The structure for the baseline model can be summarized as follows:
```text
[Linear(in_features=in_dim, out_features=hidden_dim, bias=True),
Linear(in_features=hidden_dim, out_features=out_dim, bias=True)];
```
where `out_dim = 1`.

### Other Models

1. xxx
2. xxx

### Variants

1. xxx

<br>

## Experiment

For metrics, we use MAE (Mean Absolute Error) to evaluate the model performance.

xxx

<br>
