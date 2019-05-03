"""
Original author: YJ Choe (yjchoe33@gmail.com).
"""

import tensorflow as tf
from tqdm import tqdm
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
import numpy as np
import pickle as pkl
import os


class GRUToyModel:
    """TF graph builder for the CudnnLSTM model."""

    def __init__(self, input_size=2,
                 num_layers=2, num_units=64, direction="unidirectional",
                 learning_rate=0.001, dropout=0.2, seed=0, is_training=True, model=0):
        """Initialize parameters and the TF computation graph."""

        """
        model parameters
        """
        self.input_size = input_size
        self.num_layers = num_layers
        self.num_units = num_units
        self.direction = direction
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.seed = seed
        self.model = model

        self.model_name = "toy-gru-{}-{}-{}-{}-{}".format(
            self.num_layers, self.num_units, self.direction,
            self.learning_rate, self.dropout)
        self.save_path = "./checkpoints/{}.ckpt".format(self.model_name)
        self.pickle_path = "./checkpoints/{}.pickle".format(self.model_name)

        # running TF sessions
        self.is_training = is_training
        self.saver = None
        self.sess = None

        tf.set_random_seed(self.seed)

        """
        TF graph construction
        """
        # [time_len, batch_size, input_size]
        self.inputs = tf.placeholder(tf.float32,
                                     shape=[None, None, self.input_size],
                                     name="input_placeholder")

        if model == 0:  # CudnnGRU
            self.gru = tf.contrib.cudnn_rnn.CudnnGRU(
                self.num_layers,
                self.num_units,
                direction=self.direction,
                dropout=self.dropout if is_training else 0.,
            )
            # outputs: [time_len, batch_size, num_units]
            # output_states: ([time_len, batch_size, num_units], [time_len, batch_size, num_units])
            self.outputs, self.output_states = self.gru(
                self.inputs,
                initial_state=None,
                training=True
            )
        else:  # GRUBlockCell or GRUCell
            with tf.variable_scope("cudnn_gru"):
                if not self.is_training:
                    # [num_layers, 2, batch_size, num_units]
                    self.initial_state = tf.placeholder(tf.float32,
                                                        shape=[self.num_layers, 1, None, self.num_units],
                                                        name="state_placeholder")
                    self.initial_cell_state = tuple([
                        self.initial_state[idx, 0] for idx in range(self.num_layers)
                    ])

                # Create cells
                if model == 1:
                    single_cell = lambda: tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(
                        self.num_units, reuse=tf.get_variable_scope().reuse)
                else:
                    single_cell = lambda: tf.nn.rnn_cell.GRUCell(
                        self.num_units, reuse=tf.get_variable_scope().reuse)
                self.cell = tf.nn.rnn_cell.MultiRNNCell([single_cell() for _ in range(self.num_layers)])

                # Run GRU
                self.outputs, self.output_states = tf.nn.dynamic_rnn(
                    self.cell,
                    self.inputs,
                    dtype=tf.float32,
                    time_major=True,
                    initial_state=None if self.is_training else self.initial_cell_state
                )
                self.output_state = tf.stack([
                    tf.stack([
                        state_tuple[0], state_tuple[1]
                    ]) for state_tuple in self.output_states
                ], name="output_state")

        # [time_len, batch_size, 4]
        self.logits = tf.layers.dense(self.outputs, 4, name="logits")
        # [time_len, batch_size, 2]
        self.predictions = tf.stack([tf.atan2(self.logits[:, :, 2*idx], self.logits[:, :, 2*idx+1]) for idx in range(2)], axis=2, name="predictions")

        # [time_len, batch_size, 2]
        self.labels = tf.placeholder(tf.float32, shape=[None, None, 2], name="label_placeholder")
        # [time_len, batch_size, 2]
        self.difference = self.predictions - self.labels
        # float
        self.loss = tf.reduce_mean(tf.abs(tf.atan2(tf.sin(self.difference), tf.cos(self.difference))), name="loss")

        if self.is_training:
            self.optimizer = \
                tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

        return

    def train(self, inputs_, inputs_valid_, labels_, labels_valid_,
              batch_size, num_epochs):
        assert self.is_training, \
            "train(): model not initialized in training mode"

        self.saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
        )) as sess:
            sess.run(tf.global_variables_initializer())

            # Restore
            if self.model == 0:
                if os.path.isfile(self.save_path):
                    self.saver.restore(sess, self.save_path)
                    print("========Model restored from {}========".format(
                        self.save_path))
            elif self.model == 2:
                if os.path.isfile(self.pickle_path):
                    self.restore_weights(sess)
                    print("--------Model restored from {}========".format(self.pickle_path))


            print("========Training CudnnLSTM with "
                  "{} layers and {} units=======".format(self.num_layers,
                                                         self.num_units))
            n_train = len(labels_)
            for epoch in range(num_epochs):
                print("Epoch {}:".format(epoch))
                for batch in tqdm(range(n_train // batch_size)):
                    current = batch * batch_size
                    _ = sess.run(
                        [self.train_op],
                        feed_dict={
                            self.inputs:
                                inputs_[:, current:current+batch_size, :],
                            self.labels:
                                labels_[:, current:current+batch_size, :]
                        }
                    )
                # monitor per epoch
                train_loss_ = sess.run(
                    self.loss, feed_dict={self.inputs: inputs_,
                                          self.labels: labels_})
                valid_loss_ = sess.run(
                    self.loss, feed_dict={self.inputs: inputs_valid_,
                                          self.labels: labels_valid_})
                print("\ttrain loss: {:.5f}".format(train_loss_))
                print("\tvalid loss: {:.5f}".format(valid_loss_))
                if (epoch+1) % 10 == 0:
                    if self.model == 0:
                        self.saver.save(sess, self.save_path)
                    elif self.model == 2:
                        self.save_weights(sess)

            print("========Finished training! "
                  "(Model saved in {})========".format(self.save_path))
        return

    def eval(self, inputs_test_, labels_test_):
        assert not self.is_training, \
            "eval(): model initialized in training mode"

        self.saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
        )) as sess:
            for var in tf.trainable_variables():
                print(var)

            sess.run(tf.global_variables_initializer())
            if self.model == 1:
                #self.restore_weights(sess)
                self.saver.restore(sess, self.save_path)
                print("========Model restored from {}========".format(
                    self.save_path))
            elif self.model == 2:
                self.restore_weights(sess)
                print("--------Model restored from {}========".format(self.pickle_path))

            state = np.zeros((self.num_layers, 1, inputs_test_.shape[1], self.num_units))
            test_loss_ = sess.run(
                self.loss,
                feed_dict={self.inputs: inputs_test_,
                           self.labels: labels_test_,
                           self.initial_state: state}
            )
            print("\teval loss: {:.5f}".format(test_loss_))

        return test_loss_

    def export(self):
        assert not self.is_training, \
            "export(): model initialized in training mode"

        self.saver = tf.train.Saver()

        # Define input/output names
        input_node_names = ["input_placeholder", "cudnn_gru/state_placeholder"]
        output_node_names = ["predictions", "cudnn_gru/output_state"]
        output_node_names_str = ",".join(output_node_names)

        # Hard-coded path values
        dir = "exports"
        graph_name = "toy"
        pbtxt = dir + "/" + graph_name + "_graph.pbtxt"
        chkp = dir + "/" + graph_name + ".chkp"

        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
        )) as sess:
            self.restore_weights(sess)

            tf.train.write_graph(sess.graph_def, dir, graph_name + "_graph.pbtxt")
            self.saver.save(sess, chkp)
            freeze_graph.freeze_graph(pbtxt,  # input_graph
                                      None,  # input_saver
                                      False,  # input_binary
                                      chkp,  # input_checkpoint
                                      output_node_names_str,  # output_node_names
                                      "save/restore_all",
                                      "save/Const:0",
                                      dir + "/" + "frozen_" + graph_name + ".bytes",
                                      True,
                                      "")

            input_graph_def = tf.GraphDef()
            with tf.gfile.Open(dir + "/frozen_" + graph_name + ".bytes", "rb") as f:
                input_graph_def.ParseFromString(f.read())

            output_graph_def = optimize_for_inference_lib.optimize_for_inference(
                input_graph_def, input_node_names, output_node_names,
                tf.float32.as_datatype_enum)

            with tf.gfile.FastGFile(dir + "/opt_" + graph_name + ".bytes", "wb") as f:
                f.write(output_graph_def.SerializeToString())

            print("========Model exported!========")

    def export_weights(self):
        """
        Exports trainable weights in the graph into a file.
        Can be called without a session.
        """
        assert not self.is_training, \
            "export(): model initialized in training mode"

        self.saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
        )) as sess:
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, self.save_path)
            print("========Model restored from {}========".format(self.save_path))
            self.save_weights(sess)

        print("========Model saved in {}========".format(self.pickle_path))

    def save_weights(self, sess):
        """
        Exports trainable weights in the graph into a file.
        """
        weights = {}
        for var in tf.trainable_variables():
            # Transform names from LSTMBlockCell -> LSTMCell
            var_name = var.name.replace("cudnn_compatible_", "")
            weights[var_name] = np.array(var.eval(sess), dtype=np.float32)

        pkl.dump(weights, open(self.pickle_path, "wb"))

    def restore_weights(self, sess):
        """
        Restores trainable weights from a file into the current graph.
        """

        weights = pkl.load(open(self.pickle_path, "rb"))
        for var_name in weights:
            var = sess.graph.get_tensor_by_name(var_name)
            sess.run(tf.assign(var, weights[var_name]))
