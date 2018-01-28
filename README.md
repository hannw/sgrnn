# sgrnn

## Synthetic Gradient for Recurrent Neural Networks

![Synthetic Gradient for RNN](img/sgrnn.gif?raw=true)

This repo contains a tensorflow implementation of the synthetic gradient for recurrent neural network architecture.




The data required to run the model is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:
```bash
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

```
To run the RNN model,
```bash
$ python sgrnn/main.py --model=test --data_path=simple-examples/data/ \
    --num_gpus=0 --rnn_mode=BASIC --save_path=/tmp/sgrnn
```

Reference:
- [Decoupled Neural Interfaces using Synthetic Gradients](https://arxiv.org/abs/1608.05343) 
- [Understanding Synthetic Gradients and Decoupled Neural Interfaces](https://arxiv.org/abs/1703.00522)
- [nitarshan's implementation of synthetic gradient for MLP](https://github.com/nitarshan/decoupled-neural-interfaces)
- [Tensorflow PTB tutorial](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb)