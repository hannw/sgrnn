# sgrnn

## Synthetic Gradient for Recurrent Neural Networks


This repo is a tensorflow implementation of the synthetic gradient, or DNI, for
recurrent neural network (RNN). The architecture contains a multilayer LSTM RNN that
is used for language modeling to do word-level prediction. For a detailed description of how synthetic gradient is applied to train this architecture, check out the blog post [here](https://hannw.github.io/posts/synthetic-gradient-rnn).


The data required to run the model is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:
```bash
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
```

To run the RNN model,
```bash
$ python sgrnn/main.py --model=small --data_path=simple-examples/data/ \
    --num_gpus=0 --rnn_mode=BASIC --save_path=/tmp/sgrnn
```

Reference:
- [Decoupled Neural Interfaces using Synthetic Gradients](https://arxiv.org/abs/1608.05343) 
- [Understanding Synthetic Gradients and Decoupled Neural Interfaces](https://arxiv.org/abs/1703.00522)
- [nitarshan's implementation of synthetic gradient for MLP](https://github.com/nitarshan/decoupled-neural-interfaces)
- [Tensorflow PTB tutorial](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb)