"""
A copy of AlphaZero's model.py, (unsuccessfully) converted to Tensorflow v2 
and with only the inference components preserved.
"""
import os

import numpy as np
import tensorflow as tf


class Model(object):

  def __init__(self, session, saver, path):
    """Init a model. Use build_model, from_checkpoint or from_graph instead."""
    self._session = session
    self._saver = saver
    self._path = path

    def get_var(name):
      return self._session.graph.get_tensor_by_name(name + ":0")

    self._input = get_var("input")
    self._legals_mask = get_var("legals_mask")
    self._training = get_var("training")
    self._value_out = get_var("value_out")
    self._policy_softmax = get_var("policy_softmax")

  @classmethod
  def from_checkpoint(cls, checkpoint, path=None):
    """Load a model from a checkpoint."""
    model = cls.from_graph(checkpoint, path)
    model.load_checkpoint(checkpoint)
    return model

  @classmethod
  def from_graph(cls, metagraph, path=None):
    """Load only the model from a graph or checkpoint."""
    if not os.path.exists(metagraph):
      metagraph += ".meta"
    if not path:
      path = os.path.dirname(metagraph)
    g = tf.Graph()  # Allow multiple independent models and graphs.
    with g.as_default():
      saver = tf.compat.v1.train.import_meta_graph(metagraph)
    session = tf.compat.v1.Session(graph=g)
    session.__enter__()
    session.run("init_all_vars_op")
    return cls(session, saver, path)

  def __del__(self):
    if hasattr(self, "_session") and self._session:
      self._session.close()

  def inference(self, observation, legals_mask):
    return self._session.run(
        [self._value_out, self._policy_softmax],
        feed_dict={self._input: np.array(observation, dtype=np.float32),
                   self._legals_mask: np.array(legals_mask, dtype=np.bool),
                   self._training: False})

  def load_checkpoint(self, path):
    # a failed attempt at converting the checkpoint to be loadable in Tensorflow v2
    # https://www.tensorflow.org/guide/migrate/migrating_checkpoints#checkpoint-conversion
    # tf.disable_v2_behavior()
    # vars = {}
    # reader = tf.train.load_checkpoint(path)
    # dtypes = reader.get_variable_to_dtype_map()
    # for key in dtypes.keys():
    #   vars[key] = tf.Variable(reader.get_tensor(key))
    # ckpt = tf.train.Checkpoint(vars=vars)
    # ckpt.save('converted-tf1-to-tf2')

    return self._saver.restore(self._session, path)
