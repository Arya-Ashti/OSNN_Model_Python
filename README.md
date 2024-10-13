# OSNN_Model_Python
This repository contains a python implemetation of the OSNN Algorithm proposed by Soares et al. in their paper "OSNN: An online semisupervised neural network for nonstationary data streams". The objective of this project was to perform an analysis of supervised learning and semi-supervised learning for dealing with verification latency in the context of Just-in-Time Software Defect Prediction using the OSNN model.

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
The same algorithm is used for both, the supervised and semi-supervised learning models.

A Radial Basis Function Network (RBFN) was trained using only labelled incoming instances by filtering out any unlaballed data. The model is optimised using the Newton Rhapson update method and a self-learning learning rate.

The semi-supervised approach allows unlabelled instances to remain in the set. The model utilises these instances using pseudo-labeling to exploit the underlying structure of the unlabeled data for better predicative performance.

## Installation

