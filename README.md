# COP (Predict colonization outcomes for complex microbial communities)

This is a Pytorch and Sklearn implementation of COP, as described in our paper:

Wu, L., Wang, X.W., Tao, Z., Wang, T., Zuo, W., Zeng, Y., Liu, Y.Y. and Dai, L.. [Data-driven prediction of colonization outcomes for complex microbial communities]. bioRxiv, pp.2023-03 (2023).

We have tested this code for Python 3.8.13 and R 4.1.2.

<p align="center">
  <img src="papers/workflow.png" alt="demo" width="800" height="470" style="display: block; margin: 0 auto;">
</p>


## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [Data type for DKI](#Data-type-for-DKI)
- [How the use the DKI framework](#How-the-use-the-DKI-framework)

# Overview

Previous studies suggested that microbial communities harbor keystone species whose removal can cause a dramatic shift in microbiome structure and functioning. Yet, an efficient method to systematically identify keystone species in microbial communities is still lacking. This is mainly due to our limited knowledge of microbial dynamics and the experimental and ethical difficulties of manipulating microbial communities. Here, we propose a Data-driven Keystone species Identification (DKI) framework based on deep learning to resolve this challenge. Our key idea is to implicitly learn the assembly rules of microbial communities from a particular habitat by training a deep learning model using microbiome samples collected from this habitat. The well-trained deep learning model enables us to quantify the community-specific keystoneness of each species in any microbiome sample from this habitat by conducting a thought experiment on species removal. We systematically validated this DKI framework using synthetic data generated from a classical population dynamics model in community ecology. We then applied DKI to analyze human gut, oral microbiome, soil, and coral microbiome data. We found that those taxa with high median keystoneness across different communities display strong community specificity, and many of them have been reported as keystone taxa in literature. The presented DKI framework demonstrates the power of machine learning in tackling a fundamental problem in community ecology, paving the way for the data-driven management of complex microbial communities.


# Repo Contents
(1) A synthetic dataset to test the Data-driven Keystone species Identification (DKI) framework.

(2) Python code to predict the species composition using species assemblage (cNODE2) and R code to compute keystoneness.

(3) Predicted species composition after removing each present species in each sample.

# Data type for DKI
## (1) Ptrain.csv: matrix of taxanomic profile of size N*M, where N is the number of taxa and M is the sample size (without header).


## (2) Thought experiment: thought experiemt was realized by removing each present species in each sample. This will generated three data type.
## This repository contains:
(1) Synthetic dataset to test the COP (classification and regression) generated by gLV population dyanmics.

(2) R codes to generated synthetic dataset.

(3) Python codes of NODE (Pytorch) and other two COP methods (Sklearn).

(4) Python code to calcualte colonization impact.



