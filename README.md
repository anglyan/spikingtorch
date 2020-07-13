# spikingtorch

A pytorch implementation of spiking neural networks and backpropagation
through spikes for [leaky integrate and fire](https://en.wikipedia.org/wiki/Biological_neuron_model#Leaky_integrate-and-fire) neurons.

The example code in this repo implements backpropagation through spikes
using [PyTorch](https://pytorch.org/) and it implements the methodology
and reproduces the results presented in the paper:

[A. Yanguas-Gil, Coarse scale representation of spiking neural networks:
backpropagation through spikes and application to neuromorphic
hardware](https://icons.ornl.gov/)

Running:

```bash
python3 spikingexamples.py randomfilename
```

Will train a shallow spiking network agains MNIST using 8-long spike trains
for 10 epochs and using an MSE cost function. You can check all the possible
options using

```bash
python3 spikingexamples.py --help
```

The file [spikingnet.py]("./spikingnet.py") contains the core of the
implementation. You should check the paper to understand the assumptions that
go into the two recurrent implementations of leaky integrate and fire
neurons used in this work.
