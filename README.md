# edge-ml
Quantized Tensorflow Estimator for Google EdgeTpu Accelerator, Dev Board and Intel Neural Compute Stick 2

## Getting Started
This repository contains the notebooks and py related to:  

### Prerequisites
To execute the notebooks, you have to download CIFAR10 (https://www.cs.toronto.edu/~kriz/cifar.html) and store the batches in ./datasets/cifar-10/ 
You also need to create the following forlder (./models/Tensor_CIFAR_Sparse_model/) to store the saved_model and Tensorboard data. 
You also have to install the usual suspects: Tensorflow, pickle, numpy, matplotlib...

### Installing
To run the two py files, you need both accelerator devices and a fresh install of the related environments (OpenVino and Edge-tpu).

## Authors

* **Nicolas Maquaire** - *Initial work* - [NicMaq](https://github.com/NicMaq/edge-ml)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details