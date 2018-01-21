from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

import sgrnn.reader as reader
# import util

from tensorflow.python.client import device_lib

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
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
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.num_unroll = num_unroll = config.num_unroll
    self.output_size = config.vocab_size
    batch = reader.pdb_state_saver(
      raw_data=data, batch_size=batch_size,
      num_steps=num_steps, init_states=init_states,
      num_unroll=num_unroll, num_threads=3, capacity=1000,
      allow_small_batch=False)
    self.input_data = batch.sequences['x']
    self.next_input_data = batch.sequences['next_x']
    self.targets = batch.sequences['y']
    self.next_targets = batch.sequences['next_y']
    self.state_saver = batch
    self.length = batch.length
    self.sequence = batch.sequence
    self.sequence_count = batch.sequence_count


class PTBModel(object):
  """The PTB model."""

  def __init__(self, config, is_training):
    self._config = config
    self._is_training = is_training
    self._cell = self.make_cell()

    self._is_done = None

  @property
  def cell(self):
    return self._cell

  @property
  def config(self):
    return self._config

  @property
  def zero_state(self):
    init_states = self._cell.zero_state(batch_size=1, dtype=tf.float32)
    def _downrank_zerostates(zs):
      if isinstance(zs, tuple):
        return tuple([_downrank_zerostates(s) for s in zs])
      else:
        zs = tf.squeeze(zs, axis=0)
        return zs
    init_states = _downrank_zerostates(init_states)
    return init_states

  @property
  def state_name(self):
    init_states = self.zero_state
    i = 0
    def gen_state_name(zs):
      nonlocal i
      if isinstance(zs, tuple):
        return tuple([gen_state_name(s) for s in zs])
      else:
        name = 'state_{}'.format(i)
        i += 1
        return name
    return gen_state_name(init_states)

  @property
  def sequence_is_done(self):
    return self._is_done

  @property
  def is_training(self):
    return self._is_training

  def build_graph(self, input_):
    self._input = input_
    self.batch_size = input_.batch_size
    self.num_steps = input_.num_steps
    self.num_unroll = input_.num_unroll
    self.state_saver = input_.state_saver
    self.sequence_length = input_.length
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

    logits, final_state, sg, next_sg = self._build_rnn_graph_lstm(
      inputs, next_inputs, self.sequence_length)

    # Use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        input_.targets,
        tf.ones([self.batch_size, self.num_unroll], dtype=data_type()),
        average_across_timesteps=False,
        average_across_batch=True)

    tvars = tf.trainable_variables()

    grad = self.gradient(loss, tvars, next_sg, final_state)
    grad_var = list(zip(grad, tvars))

    sg_target = self.sg_target(loss, tvars, next_sg, final_state)
    sg_loss = tf.losses.mean_squared_error(labels=sg_target, predictions=sg)
    sg_grad = tf.gradient(ys=sg_loss, xs=tvars)


    # Update the cost
    self._cost = tf.reduce_sum(loss)
    self._final_state = final_state

    self._lr = tf.Variable(0.0, trainable=False)
    grads, _ = tf.clip_by_global_norm(grad,
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

  def gradient(self, loss, tvars, next_sg, final_state):
    grad_local = tf.gradients(ys=loss, xs=tvars, grad_ys=None,
                  name='local_gradients')
    grad_sg = tf.gradients(
      ys=final_state, xs=tvars, grad_ys=next_sg,
      name='synthetic_gradients')
    grad = grad_local + grad_sg
    return grad

  def sg_target(self, loss, tvars, next_sg, final_state):
    sg_target = (tf.gradients(ys=loss, xs=self.initial_state)
      + tf.gradients(
        ys=final_state, xs=self.initial_state, grad_ys=next_sg))
    return tf.stop_gradient(sg_target, name='sg_target')

  def _get_lstm_cell(self, config, is_training):
    if config.rnn_mode == BASIC:
      return tf.contrib.rnn.BasicLSTMCell(
          config.hidden_size, forget_bias=0.0, state_is_tuple=True,
          reuse=not is_training)
    if config.rnn_mode == BLOCK:
      return tf.contrib.rnn.LSTMBlockCell(
          config.hidden_size, forget_bias=0.0)
    raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

  def _make_single_cell(self):
    cell = self._get_lstm_cell(self.config, self.is_training)
    if self.is_training and self.config.keep_prob < 1:
      cell = tf.contrib.rnn.DropoutWrapper(
        cell, output_keep_prob=self.config.keep_prob)
    return cell

  def make_cell(self):
    cell = tf.contrib.rnn.MultiRNNCell(
      [self._make_single_cell() for _ in range(self.config.num_layers)],
      state_is_tuple=True)
    n_hidden = self.config.hidden_size
    total_hidden_size = n_hidden * 2 * self.config.num_layers
    output_size = self.config.vocab_size
    cell = tf.contrib.rnn.OutputProjectionWrapper(
      cell, output_size + total_hidden_size)
    return cell

  def _build_rnn_graph_lstm(
        self, inputs, next_inputs, sequence_length):

    inputs = tf.unstack(inputs, num=self.num_unroll, axis=1)
    next_inputs = tf.unstack(next_inputs, num=self.num_unroll, axis=1)

    with tf.variable_scope('RNN') as scope:
      outputs, final_state = tf.nn.static_state_saving_rnn(
        cell=self.cell,
        inputs=inputs,
        state_saver=self.state_saver,
        state_name=self.state_name,
        sequence_length=sequence_length)

    with tf.variable_scope(scope, reuse=True):
      next_outputs, next_final_state = tf.nn.static_rnn(
          cell=self.cell,
          inputs=next_inputs,
          initial_state=final_state,
          sequence_length=sequence_length)

    output_size = self.config.vocab_size

    synthetic_gradient = tf.slice(
      outputs[0], begin=[0, output_size], size=[-1, -1])
    next_synthetic_gradient = tf.slice(
      next_outputs[0], begin=[0, output_size], size=[-1, -1])

    assert False

    with tf.variable_scope('logits'):
      logits = [tf.slice(output, begin=[0, 0], size=[-1, -1]) for output in outputs]
      logits = [tf.expand_dims(logit, axis=1) for logit in logits]
      logits = tf.concat(logits, axis=1)

    return logits, final_state, synthetic_gradient, next_synthetic_gradient

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  def export_ops(self, name):
    """Exports ops to collections."""
    self._name = name
    ops = {util.with_prefix(self._name, "cost"): self._cost}
    if self._is_training:
      ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
      if self._rnn_params:
        ops.update(rnn_params=self._rnn_params)
    for name, op in ops.items():
      tf.add_to_collection(name, op)
    self._initial_state_name = util.with_prefix(self._name, "initial")
    self._final_state_name = util.with_prefix(self._name, "final")
    util.export_state_tuples(self._initial_state, self._initial_state_name)
    util.export_state_tuples(self._final_state, self._final_state_name)

  def import_ops(self):
    """Imports ops from collections."""
    if self._is_training:
      self._train_op = tf.get_collection_ref("train_op")[0]
      self._lr = tf.get_collection_ref("lr")[0]
      self._new_lr = tf.get_collection_ref("new_lr")[0]
      self._lr_update = tf.get_collection_ref("lr_update")[0]
      rnn_params = tf.get_collection_ref("rnn_params")
      if self._cell and rnn_params:
        params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
            self._cell,
            self._cell.params_to_canonical,
            self._cell.canonical_to_params,
            rnn_params,
            base_variable_scope="Model/RNN")
        tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
    self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
    num_replicas = FLAGS.num_gpus if self._name == "Train" else 1
    self._initial_state = util.import_state_tuples(
        self._initial_state, self._initial_state_name, num_replicas)
    self._final_state = util.import_state_tuples(
        self._final_state, self._final_state_name, num_replicas)

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def train_sg_op(self):
    return self._train_sg_op

  @property
  def initial_state_name(self):
    return self._initial_state_name

  @property
  def final_state_name(self):
    return self._final_state_name


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
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK
  num_unroll = 3


def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
             (time.time() - start_time)))

  return np.exp(costs / iters)


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
  gpus = [
      x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"
  ]
  if FLAGS.num_gpus > len(gpus):
    raise ValueError(
        "Your machine has only %d gpus "
        "which is less than the requested --num_gpus=%d."
        % (len(gpus), FLAGS.num_gpus))

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(
          config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config,
                         input_=test_input)

    models = {"Train": m, "Valid": mvalid, "Test": mtest}
    for name, model in models.items():
      model.export_ops(name)
    metagraph = tf.train.export_meta_graph()
    if tf.__version__ < "1.1.0" and FLAGS.num_gpus > 1:
      raise ValueError("num_gpus > 1 is not supported for TensorFlow versions "
                       "below 1.1.0")
    soft_placement = False
    if FLAGS.num_gpus > 1:
      soft_placement = True
      util.auto_parallel(metagraph, m)

  with tf.Graph().as_default():
    tf.train.import_meta_graph(metagraph)
    for model in models.values():
      model.import_ops()
    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
    with sv.managed_session(config=config_proto) as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                     verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      test_perplexity = run_epoch(session, mtest)
      print("Test Perplexity: %.3f" % test_perplexity)

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
  tf.app.run()