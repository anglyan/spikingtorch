# spikingtorch

Training spiking neural networks using Pytorch

## About

`spikingtorch` is a lightweight package for training deep neural
networks using Pytorch. `spikingtorch` includes encoders that transform
standard ML datasets into spike trains, and decoders that transform
the output spikes into values that can be used with loss functions in
Pytorch.

`spikingtorch` implements  spiking neural networks and backpropagation
through spikes for [leaky integrate and fire](https://en.wikipedia.org/wiki/Biological_neuron_model#Leaky_integrate-and-fire) neurons.

In addition to the Python package, this repository also
 implements the methodology
and reproduces the results presented in the paper:

[A. Yanguas-Gil, Coarse scale representation of spiking neural networks:
backpropagation through spikes and application to neuromorphic
hardware, arXiv:2007.06176](https://arxiv.org/abs/2007.06176)

## Status

`spikingtorch` is in active development, with more neuron models coming up
soon.

## Quick install

Through pypi:

```
pip install spikingtorch
```

## Acknowledgements

* Threadwork, U.S. Department of Energy Office of Science, 
  Microelectronics Program.

The original implementation was based on reseach funded through Argonne's Laboratory Directed Research
and Development program.

## Publications

Spikingtorch's backpropagation approach is based on the following work:

[A. Yanguas-Gil, Coarse scale representation of spiking neural networks:
backpropagation through spikes and application to neuromorphic
hardware, arXiv:2007.06176](https://arxiv.org/abs/2007.06176)


## Copyright and license

Copyright © 2023, UChicago Argonne, LLC

spikingtorch is distributed under the terms of BSD License. See 
[LICENSE](https://github.com/anglyan/spikingtorch/blob/master/LICENSE.md)

