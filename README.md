# MS-HyAttNet
This repository provides code for the Multiscale Hybrid Attention Network (MS-HyAttNet) proposed in the paper: MS-HyAttNet: Multiscale Hybrid Attention Network for Motor Imagery EEG Decoding

MS-HyAttNet model is composed of four main modules: the feature extraction module (FE), Squeeze-and-excitation attention module (SEA), multi-head self-attention module (MSA), and temporal convolutional network (TCN).

![The overall architecture of MS-HyAttNet](https://github.com/xiaorong777/MS-HyAttNet/blob/main/MS-HyAttNet.png)

## Dataset
The BCI Competition IV-2a and BCI Competition IV-2b dataset needs to be downloaded. The dataset can be downloaded from [here](http://www.bbci.de/competition/iv/).
## Development environment
Models were trained and tested by a single GPU, [Nvidia GTX 3090 24GB](https://www.nvidia.com/en-me/geforce/graphics-cards/30-series/) (Driver Version: 510.108.03, CUDA 11.6), using Python 3.10.4 with PyTorch framework. Anaconda 3 was used on Ubuntu 18.04.6 LTS. The following packages are required:
- PyTorch: 1.13.1
- TorchVision: 0.14.1
- matplotlib: 3.9.2
- NumPy: 1.23.5
- scikit-learn: 1.5.2
- SciPy: 1.14.1

Refer to the documentation for more detailed dependency package information: [requirement.txt](https://github.com/xiaorong777/MS-HyAttNet/blob/main/requirements.txt)


