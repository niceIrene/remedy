# Mitigating Subgroup Unfairness in Machine Learning Classifiers: A Data-Driven Approach

Imbalanced sample collection can lead to unfairness in learned models due to historical biases and a lack of control over data collection. Through the introduction of “Implicit Biased Set (IBS)", we propose an efficient pre-processing algorithm to identify IBS and then propose data remedy techniques to balance the data collection in IBS.

This repository contains code for the Identification Algorithms, Data Remedy Methods, Divexplorer metrics and ML model settings and a demo notebook to run both effectiveness and efficency experiments in the paper.

## Prerequisites
To install the package and prepare for use, run:
<pre><code>git clone https://github.com/niceIrene/remedy.git

pip install -r requirements.txt
</code></pre>

The following python packages are required to run the code: <b>divexplorer</b>, pandas, sklearn, numpy, sympy.

## Demo
For a demonstration of our working code and results, use this: [COMPAS Dataset Effectiveness Demo](./Demo_Effectiveness_COMPAS.ipynb).

It is also possible to view the individual algorithms for:

[Data Remedy Methods](./remedy.py)

[Identification Algorithms](./identification.py)

[ML model settings and divexplorer](./models.py)

## Datasets
Adult Dataset: https://archive.ics.uci.edu/ml/datasets/adult. 

COMPAS Dataset: https://github.com/propublica/compas-analysis. 

Law School Dataset: https://www.kaggle.com/datasets/danofer/law-school-admissions-bar-passage   

