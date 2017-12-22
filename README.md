word-rnn-tensorflow
===================

[![Build Status](https://travis-ci.org/karantan/word-rnn-tensorflow.svg?branch=master)](https://travis-ci.org/karantan/word-rnn-tensorflow)

Multi-layer Recurrent Neural Networks (LSTM, RNN) for word-level language models in Python using TensorFlow.

Mostly reused code from https://github.com/sherjilozair/char-rnn-tensorflow which was inspired from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).


Installation Prerequisites
--------------------------

Technically you can run this on CPU but it takes forever. We strongly suggest
using GPU for training the model. So first you need is a good GPU
(e.g. GeForce GTX 1080). Then you need to have installed the following
packages:

- [Latest CUDA toolki](https://developer.nvidia.com/cuda-toolkit)
- [Conda](https://conda.io/docs/)


Installation
------------

Use the following command to install word-rnn-tensorflow text model generation:

```bash
git clone https://github.com/karantan/word-rnn-tensorflow
cd word-rnn-tensorflow
conda env create -f environment.yml
source activate rnn-tf
pip install -e .

```

For installing development dependencies type:

```bash
pip install -e .[dev]

```


Train the model
---------------

Before you start training the model you should review the default configuration
in the `config.yml` file.

Use the following command to train the model:

```bash
TODO

```

For more options run with `--help` option:

```bash
TODO
```

Before you start training the model please check the GPU usage by running the
following command:

```bash
nvidia-smi

```

After running, a directory called `/logs` is generated with logs. These logs
can be used with tensorboard to display various graphs. To check the
tensorboard, run:

```
$ tensorboard --logdir=./logs --port [PORT]
```


Generate text
-------------

Use the following command to generate text:

```bash
TODO

```

For more options run with `--help` option:

```bash
TODO

```


Contribute
----------

- Issue Tracker: github.com/niteoweb/kai/issues
- Source Code: github.com/niteoweb/kai

Support
-------

If you are having issues, please let us know.
