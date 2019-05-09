# Quantized Estimator for making inference on the edge.
Using the Google Coral Accelerator, the Google Coral Dev Board or the Intel Neural Compute Stick 2.

This repository contains all files required to create a deep neural network in Tensorflow, to port this model on a Raspberry Pi and to make the inference using either the Google Edge-tpu accelerator or the Intel Neural Compute Stick v2 to increase performance. 

We evaluate both environments to understand the depth of the knowledge required to use these devices. The goal is not to search for the best performances with regards to the models, losses, metrics and architecture.  

The plan is to design a fresh Convolutional Neural Network with Tensorflow. As these chips work faster with integers or half precision floating points, we will build a model running on a low precision data-type. To do this, we will operate a quantization aware training for our model. Then, we will freeze the graph to compile it into the different formats the two accelerators support. Finally, we’ll try our model on the edge.

You can find the full repor on Medium: https://medium.com/p/21234ea7633/edit


### Prerequisites
To go through the notebooks, you have first to download CIFAR10 (https://www.cs.toronto.edu/~kriz/cifar.html) and store the batches in ./datasets/cifar-10. You also need to create the following forlder (./models/Tensor_CIFAR_Sparse_model/) to store the saved_model and Tensorboard data. You also have to install the usual suspects: Tensorflow, pickle, numpy, matplotlib... see first cell for the list of dependencies.

### Installing
To run the two py files, you need at least one of the accelerator devices and a fresh install of the related environments (OpenVino or Edge-tpu). There are plenty of good tutorials explaining how to deploy these environments.

## Authors

* **Nicolas Maquaire** - *Initial work* - [NicMaq](https://github.com/NicMaq/edge-ml)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
