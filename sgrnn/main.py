from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import copy

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

import reader
from sgrnn.model import SyntheticGradientRNN

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", '/tmp/sgrnn/',
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("num_gpus", 1,
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
flags.DEFINE_string("rnn_mode", None,
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                    "and lstm_block_cell classes.")
FLAGS = flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"


def _get_total_hidden_size(cell):
  if isinstance(cell, tuple):
    cells = nest.flatten(cell)
    sizes = [c.get_shape().as_list()[0] for c in cells]



def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, init_states, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    # self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.epoch_size = (len(data) - 1) // config.num_unroll // batch_size
    self.num_unroll = num_unroll = config.num_unroll
    self.output_size = config.vocab_size
    batch = reader.pdb_state_saver(
      raw_data=data, batch_size=batch_size,
      num_steps=num_steps, init_states=init_states,
      num_unroll=num_unroll, num_threads=3, capacity=1000,
      allow_small_batch=False, epoch=1000)
    self.input_data = batch.sequences['x']
    self.next_input_data = batch.sequences['next_x']
    self.targets = batch.sequences['y']
    self.next_targets = batch.sequences['next_y']
    self.state_saver = batch
    self.length = batch.length
    self.sequence = batch.sequence
    self.sequence_count = batch.sequence_count


class PTBModel(SyntheticGradientRNN):
  def __init__(self, config, is_training):
    super().__init__()
    self._config = config
    self._num_layers = config.num_layers
    self._is_training = is_training
    self._output_size = config.vocab_size
    self._input = None
    self._batch_size = None
    self._num_steps = None
    self._sequence_length = None
    self._is_done = None

    self._rnn_params = None
    self._final_state = None

  @property
  def is_training(self):
    return self._is_training

  @property
  def config(self):
    return self._config

  @property
  def num_layers(self):
    return self._num_layers

  @property
  def input(self):
    return self._input

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def num_steps(self):
    return self._num_steps

  @property
  def sequence_length(self):
    return self._sequence_length

  @property
  def is_done(self):
    return self._is_done

  @property
  def base_cell(self):
    if not self._base_cell:
      self._base_cell = tf.contrib.rnn.MultiRNNCell(
        [self._make_single_cell() for _ in range(self.num_layers)],
        state_is_tuple=True)
    return self._base_cell

  def _make_single_cell(self):
    cell = self._get_lstm_cell(self.config, self.is_training)
    if self.is_training and self.config.keep_prob < 1:
      cell = tf.contrib.rnn.DropoutWrapper(
        cell, output_keep_prob=self.config.keep_prob)
    return cell

  def _get_lstm_cell(self, config, is_training):
    if config.rnn_mode == BASIC:
      return tf.contrib.rnn.BasicLSTMCell(
          config.hidden_size, forget_bias=0.0, state_is_tuple=True,
          reuse=not is_training)
    if config.rnn_mode == BLOCK:
      return tf.contrib.rnn.LSTMBlockCell(
          config.hidden_size, forget_bias=0.0)
    raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

  def build_graph(self, input_):
    self._input = input_
    self._batch_size = input_.batch_size
    self._num_steps = input_.num_steps
    self._num_unroll = input_.num_unroll
    self._state_saver = input_.state_saver
    self._init_state = [self.state_saver.state(name)
                        for name in nest.flatten(self.state_name)]
    self._sequence_length = input_.length
    self._is_done = tf.equal(input_.sequence, input_.sequence_count - 1)
    size = self.config.hidden_size
    vocab_size = self.config.vocab_size

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
      next_inputs = tf.nn.embedding_lookup(embedding, input_.next_input_data)

    if self.is_training and self.config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, self.config.keep_prob)
      next_inputs = tf.nn.dropout(next_inputs, self.config.keep_prob)

    logits, final_state, sg = self.build_synthetic_gradient_rnn(
      inputs, self.sequence_length)
    next_sg = self.build_next_synthetic_gradient(final_state, next_inputs)

    self._final_state = final_state

    # Use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        input_.targets,
        tf.ones([self.batch_size, self.num_unroll], dtype=data_type()),
        average_across_timesteps=False,
        average_across_batch=True)
    loss = tf.reduce_sum(loss, axis=0, keep_dims=False)

    tvars = tf.trainable_variables()

    grad = self.gradient(loss, tvars, next_sg, final_state)

    sg_target = self.sg_target(loss, next_sg, final_state)
    sg_loss = tf.losses.mean_squared_error(labels=tf.stack(sg_target), predictions=tf.stack(sg))
    sg_grad = tf.gradients(ys=sg_loss, xs=tvars)

    # Update the cost
    self._cost = tf.reduce_sum(loss)
    self._final_state = final_state
    self._sg_cost = sg_loss

    self._lr = tf.Variable(0.0, trainable=False)
    grads, _ = tf.clip_by_global_norm(grad,
                                      self.config.max_grad_norm)
    sg_grad, _ = tf.clip_by_global_norm(sg_grad,
                                        self.config.max_grad_norm)
    optimizer = tf.train.AdamOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.train.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

    optimizer_sg = tf.train.AdamOptimizer(self._lr)
    self._train_sg_op = optimizer_sg.apply_gradients(
      grads_and_vars=zip(sg_grad, tvars),
      global_step=tf.train.get_or_create_global_step())

  @property
  def train_op(self):
    return self._train_op

  @property
  def train_sg_op(self):
    return self._train_sg_op

  @property
  def cost(self):
    return self._cost

  @property
  def sg_cost(self):
    return self._sg_cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK
  num_unroll = 5

class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK
  num_unroll = 5

class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK
  num_unroll = 5

class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 30
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 8
  vocab_size = 10000
  rnn_mode = BLOCK
  num_unroll = 3


def run_epoch(session, model, global_step, train_ops=None,
              summary_op=None, verbose=False, summary_writer=None):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
      "sg_cost": model.sg_cost
  }

  is_training = False
  if train_ops is not None:
    fetches.update(train_ops)
    is_training = True

  if summary_op is not None:
    summary_dict = {'summary': summary_op}
  else:
    summary_dict = {}

  fetches_w_summary = copy.copy(fetches)
  fetches_w_summary.update(summary_dict)

  for step in range(model.input.epoch_size):
    if is_training:
      global_step += 1
    if step % 10 == 0:
      vals = session.run(fetches_w_summary)
      summary = vals['summary']
      summary_writer.add_summary(summary, global_step)
    else:
      vals = session.run(fetches)

    cost = vals["cost"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
             (time.time() - start_time)))

  return np.exp(costs / iters), global_step


def get_config():
  """Get model config."""
  config = None
  if FLAGS.model == "small":
    config = SmallConfig()
  elif FLAGS.model == "medium":
    config = MediumConfig()
  elif FLAGS.model == "large":
    config = LargeConfig()
  elif FLAGS.model == "test":
    config = TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)
  if FLAGS.rnn_mode:
    config.rnn_mode = FLAGS.rnn_mode
  if FLAGS.num_gpus != 1 or tf.__version__ < "1.3.0" :
    config.rnn_mode = BASIC
  return config


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data

  config = get_config()

  initializer = tf.random_uniform_initializer(-config.init_scale,
                                              config.init_scale)

  with tf.name_scope("Train"):

    with tf.variable_scope("Model", reuse=None, initializer=initializer):
      m = PTBModel(is_training=True, config=config)
      train_input = PTBInput(
        config=config, data=train_data, name="TrainInput",
        init_states=m.init_state_dict)
      m.build_graph(train_input)
    tf.summary.scalar("Training Loss", m.cost)
    tf.summary.scalar("Synthetic Gradient MSE", m.sg_cost)
    tf.summary.scalar("Learning Rate", m.lr)

  with tf.name_scope("Valid"):

    with tf.variable_scope("Model", reuse=True, initializer=initializer):
      mvalid = PTBModel(is_training=False, config=config)
      valid_input = PTBInput(
        config=config, data=valid_data, name="ValidInput",
        init_states=mvalid.init_state_dict)
      mvalid.build_graph(valid_input)
    tf.summary.scalar("Validation Loss", mvalid.cost)

  summary_op = tf.summary.merge_all()

  with tf.name_scope("Test"):

    with tf.variable_scope("Model", reuse=True, initializer=initializer):
      mtest = PTBModel(is_training=False, config=config)
      test_input = PTBInput(
        config=config, data=test_data,
        init_states=mtest.init_state_dict, name="TestInput")
      mtest.build_graph(test_input)

  with tf.Session() as session:
    train_writer = tf.summary.FileWriter(FLAGS.save_path + '/train',
                                         session.graph)
    valid_writer = tf.summary.FileWriter(FLAGS.save_path + '/valid')
    print("begin training")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)

    session.run(tf.global_variables_initializer())
    global_step = 0
    for i in range(config.max_max_epoch):
      lr_decay = config.lr_decay ** max(i + 1. - config.max_epoch, 0.0)
      m.assign_lr(session, config.learning_rate * lr_decay)

      print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
      train_perplexity, global_step = run_epoch(
        session, m, global_step=global_step,
        train_ops={"train": m.train_op, "train_sg": m.train_sg_op},
        verbose=True, summary_op=summary_op, summary_writer=train_writer)
      print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
      valid_perplexity, global_step = run_epoch(session, mvalid, global_step=global_step,
                                   summary_op=summary_op, summary_writer=valid_writer)
      print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

    test_perplexity, global_step = run_epoch(session, mtest, global_step=global_step)
    print("Test Perplexity: %.3f" % test_perplexity)

    coord.request_stop()
    coord.join(threads)


if __name__ == "__main__":
  tf.app.run()