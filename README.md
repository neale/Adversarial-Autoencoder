# Adversarial-Autoencoder
An adversarial autoencoder implementation in pytorch using the WGAN with gradient penalty framework. 

There's a lot to tweak here as far as balancing the adversarial vs reconstruction loss, but this works and I'll update as I go along. 

The MNIST dataset seems to converge at around 30K steps, while CIFAR arguable doesn't output anything realistic ever. Nonetheless it starts to looks ok at around 50K steps


## MNIST - 33k steps

![output image](plots/mnist/samples_33099.jpg)


## CIFAR10 - 200k steps

![output image](plots/cifar10/samples_199999.jpg)

### Requirements

* pytorch 0.2.0
* python 3 - but 2.7 just requires some simple modifications
* matplotlib / numpy / scipy

### Usage

To train on MNIST

` python3 train.py --dataset mnist --batch_size 50 --dim 32 -- output_size 784`

To train on CIFAR10

` python3 train.py --dataset cifar10 --batch_size 64 --dim 32 -- output_size 3072`

### Acknowledgements

For the wgan-gp components I mostly used [caogang's](https://github.com/caogang/wgan-gp) nice implementation
