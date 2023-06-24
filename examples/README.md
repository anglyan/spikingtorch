# spikingtorch example

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

Examples implemented in the paper include:

  - Shallow spiking networks
  - A network with one hidden layer, all-to-all connections
  - Two convolutional networks with different depths
  - Two implementations of LeNet5, one for random spike trains and a second
    one using analog inputs to a layer of spiking neurons

The examples provide two different ways of training the networks: one using
an MSE cost function, and a second one implementing cross entropy with some
caveats. These can be selected using the `--cost` flag.

The networks
can be trained against MNIST or Fashion MNIST through the
`--dataset` flag.


The file [spikingnet.py]("./spikingnet.py") contains the core of the
implementation. The assumptions that
go into the two recurrent implementations of leaky integrate and fire
neurons used in this work are described in detail in the [paper](https://arxiv.org/abs/2007.06176).
