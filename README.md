# OSNN_Model_Python
This repository contains a python implemetation of the OSNN Algorithm proposed by Soares et al. in their paper "OSNN: An online semisupervised neural network for nonstationary data streams". The objective of this project was to perform an analysis of supervised learning and semi-supervised learning for dealing with verification latency in the context of Just-in-Time Software Defect Prediction using the OSNN model.

## Introduction
OSNN is an online semisupervised neural network designed for data streams as the network trains on every incoming instance. It is a Radial Basis Function Neural Network, utilising centers to learn patterns and regions from incoming unlabelled instances.

This project allows for semi-supervised learning by allowing unlabelled instances to appear in the dataset, and supervised learning by filtering out unlabelled instances before performing an update.

## Datasets
The datasets that will be used are extracted from a public, opensource GitHub repository made available at https://zenodo.org/record/2594681. The datasets come from opensource projects and contain information regarding a number of different metrics of each change that is committed. These metrics include
- If the change was a fix
- \# modified subsystems
- \# modified directories
- \# modified files
- Distribution of modified code
- \# lines of code added
- \# lines of code removed
- \# lines before the change
- \# developers involved
- Average time interval between crrent and last change
- \# unique changes made
- Dev experience
- Recent dev experience
- Dev experience on a subsystem

## Methods

### Supervised Learning
A Radial Basis Function Network (RBFN) was trained using fully labeled commit data. The RBFN was optimized using methods such as gradient descent and k-means clustering to determine the optimal parameters.

### Semi-supervised Learning
The semi-supervised approach utilized a small subset of labeled data along with a large pool of unlabeled data. The RBFN was extended to incorporate techniques like pseudo-labeling and graph-based methods to exploit the underlying structure of the unlabeled data.
## Installation

