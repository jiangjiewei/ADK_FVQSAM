# ADK_FVQSAM-Source: Automatic diagnosis of keratitis by integrating feature vector quantization and self-attention mechanism
# create time: 2024.7.13

# Introduction
This repository contains the source code for the Automatic Diagnosis of Keratitis using Feature Vector Quantization and Self-Attention Mechanism (ADK_FVQSAM). 
This method effectively improves the accuracy of automatic keratitis diagnosis on low-quality slit-lamp images by integrating feature vector quantization and self-attention mechanisms.

# Prerequisites
* Ubuntu: 18.04 lts
* Python 3.7.8
* Pytorch 1.6.0
* NVIDIA GPU + CUDA_10.1 CuDNN_7.5

This repository has been tested on four NVIDIA GeForce RTX 2080Ti. Configurations (e.g batch size, image patch size) may need to be changed on different platforms.

# Installation
Other packages are as follows:
* pytorch: 1.6.0 
* wheel: 0.34.2
* yaml:  0.2.5
* numpy: 1.19.1
* opencv-python: 4.6.0.66
* tourchvision: 0.7.0
* wheel:  0.34.2
* timn: 0.4.12
* scikit-image: 0.19.2
* scikit-learn: 0.23.2
* matplotlib: 3.3.1
* efficientnet-pytorch: 0.7.1
* ipython: 7.30.1
* pandas: 1.2.3
* protobuf: 3.13.0
* h5py: 3.5.0


# Install dependencies
pip install -r requirements.txt

# Usage
* The file "train.py" in /ADK_FVQSAM-Source is used for the ADK_FVQSAM model training.
* The file "test.py" in /ADK_FVQSAM-Source is used for the ADK_FVQSAM model testing.
* The file "models.py" in /ADK_FVQSAM-Source contains the ADK_FVQSAM model structure.



The expected output: print the classification probabilities for  keratitis, other corneal abnormalities (Other), and normal corneas (Normal).


* Please feel free to contact us for any questions or comments: Jiewei Jiang, E-mail: jiangjw924@126.com or XIN Yu, E-mail: xinyu2918@163.com.
