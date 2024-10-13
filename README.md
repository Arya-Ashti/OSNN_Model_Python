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

### Dataset requirements
The code in this repository is specifically designed for the dataset used in this project, which has 14 attributes per instance. If you are using a different dataset with a different number of attributes, you will need to modify certain parts of the code where the feature size is expected to be 14.

Key places to modify:
- centers_training.py, lines 48-50, 93, 123-124
- predict_function.py, lines 8, 23
- weight_update.py, line 129
- OSNN.ipynb
    - cell 2 line 6
    - cell 3, lines 28, 30, 42, 44, 58, 61, 72, 108, 111
- import_and_save.py, changes made to this file are dependant on the structure of your dataset.
- evaluation_functions.py, this will only require changes if you change the structure of the default output matrix.



## Methods
The same algorithm is used for both, the supervised and semi-supervised learning models.

A Radial Basis Function Network (RBFN) was trained using only labelled incoming instances by filtering out any unlaballed data. The model is optimised using the Newton Rhapson update method and a self-learning learning rate.

The semi-supervised approach allows unlabelled instances to remain in the set. The model utilises these instances using pseudo-labeling to exploit the underlying structure of the unlabeled data for better predicative performance.

## Installation
To run the code, you need to install the following libraries:
- NumPy
- Pandas

All .py and .ipynb files and the notebook must be in the same directory. To begin a run, open the Jupyter Notebook file and run the cells as necessary.


