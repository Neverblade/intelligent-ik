"""
Original author: YJ Choe (yjchoe33@gmail.com).
"""

import os
import pickle as pkl

import data_processing
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tqdm import tqdm


class LSTMModel:
    """TF graph builder for the CudnnLSTM model."""

    def __init__(self, input_size=2, output_size=39,
                 num_layers=2, num_units=64, direction="unidirectional",
                 learning_rate=0.001, dropout=0.2, seed=0, is_training=True, model=0):
        """Initialize parameters and the TF computation graph."""

        """
        model parameters
        """
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_units = num_units
        self.direction = direction
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.seed = seed
        self.model = model

        self.model_name = "cudnnlstm-{}-{}-{}-{}-{}".format(
            self.num_layers, self.num_units, self.direction,
            self.learning_rate, self.dropout)
        self.save_path = "./checkpoints/{}.ckpt".format(self.model_name)
        self.pickle_path = "./checkpoints/{}.pickle".format(self.model_name)

        # running TF sessions
        self.is_training = is_training
        self.saver = None
        self.sess = None

        """
        TF graph construction
        """
        # [time_len, batch_size, input_size]
        self.inputs = tf.placeholder(tf.float32,
                                     shape=[None, None, self.input_size],
                                     name="input_placeholder")
        # [time_len, batch_size, output_size]
        self.labels = tf.placeholder(tf.float32, shape=[None, None, self.output_size], name="label_placeholder")

        if model == 0:  # CudnnLSTM
            self.lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
                self.num_layers,
                self.num_units,
                direction=self.direction,
                dropout=self.dropout if is_training else 0.,
                # kernel_initializer=tf.contrib.layers.xavier_initializer()
            )

            # outputs: [time_len, batch_size, num_units]
            # output_states: ([time_len, batch_size, num_units], [time_len, batch_size, num_units])
            self.outputs, self.output_states = self.lstm(
                self.inputs,
                initial_state=None,
                training=True
            )
        else:  # LSTMBlockCell or LSTMCell
            with tf.variable_scope("cudnn_lstm"):
                if not self.is_training:
                    # [num_layers, 2, batch_size, num_units]
                    self.initial_state = tf.placeholder(tf.float32,
                                                        shape=[self.num_layers, 2, None, self.num_units],
                                                        name="state_placeholder")
                    self.initial_cell_state = tuple([
                        tf.nn.rnn_cell.LSTMStateTuple(
                            self.initial_state[idx, 0],
                            self.initial_state[idx, 1]
                        ) for idx in range(self.num_layers)
                    ])

                # Create cells
                if model == 1:
                    single_cell = lambda: tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(
                        self.num_units, reuse=tf.get_variable_scope().reuse)
                else:
                    single_cell = lambda: tf.nn.rnn_cell.LSTMCell(
                        self.num_units, reuse=tf.get_variable_scope().reuse)
                self.cell = tf.nn.rnn_cell.MultiRNNCell([single_cell() for _ in range(self.num_layers)])

                # Run LSTM
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

        self.logits = tf.layers.dense(self.outputs, self.output_size, name="logits")
        self.predicts = tf.identity(self.logits, name="predictions")
        self.loss = tf.losses.mean_squared_error(self.labels, self.logits,
                                                 reduction=tf.losses.Reduction.SUM)
        if self.is_training:
            self.optimizer = \
                tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

        return

    def train(self, data_storage, batch_size, num_epochs):
        assert self.is_training, \
            "train(): model not initialized in training mode"

        self.saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
        )) as sess:
            sess.run(tf.global_variables_initializer())

            # Restore
            if self.model == 0 or self.model == 1:
                if os.path.isfile(self.save_path + ".index"):
                    self.saver.restore(sess, self.save_path)
                    print("========Model restored from {}========".format(
                        self.save_path))
            elif self.model == 2:
                if os.path.isfile(self.pickle_path):
                    self.restore_weights(sess)
                    print("--------Model restored from {}========".format(self.pickle_path))

            # Initialize vars
            np.random.shuffle(data_storage.indices)
            n_valid = 1000
            n_train = data_storage.indices.shape[0] - n_valid
            num_batches = n_train // batch_size
            n_train = num_batches * batch_size
            ex_input, ex_label = data_storage.get_slice((0, 0, 0))
            input_batch = np.zeros((ex_input.shape[0], batch_size, ex_input.shape[1]))
            label_batch = np.zeros((ex_label.shape[0], batch_size, ex_label.shape[1]))
            n_t_label_elements = ex_label.shape[0] * n_train * ex_label.shape[1]

            # Construct indices and validation set
            t_indices, v_indices = data_storage.indices[n_valid:], data_storage.indices[:n_valid]
            v_inputs = np.zeros((ex_input.shape[0], n_valid, ex_input.shape[1]))
            v_labels = np.zeros((ex_label.shape[0], n_valid, ex_label.shape[1]))
            for i in range(n_valid):
                v_inputs[:, i, :], v_labels[:, i, :] = data_storage.get_slice(v_indices[i])
            n_v_label_elements = ex_label.shape[0] * n_valid * ex_label.shape[1]
            v_labels, v_inputs = data_processing.insert_root_motion(v_labels, v_inputs)

            print("========Training CudnnLSTM with "
                  "{} layers and {} units=======".format(self.num_layers,
                                                         self.num_units))
            for epoch in range(num_epochs):
                print("Epoch {}:".format(epoch))

                # Training
                np.random.shuffle(t_indices)
                total_train_loss = 0
                for batch in tqdm(range(num_batches)):
                    current = batch * batch_size

                    # Get batch and transfer root motion
                    for i in range(batch_size):
                        input_batch[:, i, :], label_batch[:, i, :] =\
                            data_storage.get_slice(t_indices[current + i])
                    labels, inputs =\
                        data_processing.insert_root_motion(label_batch, input_batch)
                    inputs, labels = data_processing.augment_xz(inputs, labels)
                    _, loss = sess.run(
                        [self.train_op, self.loss],
                        feed_dict={
                            self.inputs: inputs,
                            self.labels: labels
                        }
                    )
                    total_train_loss += loss
                train_loss = total_train_loss / n_t_label_elements
                print("\ttrain loss: {:.5f}".format(train_loss))

                # Validation
                valid_loss_ = sess.run(
                    self.loss, feed_dict={self.inputs: v_inputs,
                                          self.labels: v_labels}) / n_v_label_elements
                print("\tvalid loss: {:.5f}".format(valid_loss_))

                if (epoch+1) % 1 == 0:
                    if self.model == 0:
                        self.saver.save(sess, self.save_path)
                    elif self.model == 2:
                        self.save_weights(sess)

            print("========Finished training! "
                  "(Model saved in {})========".format(self.save_path))
        return

    def predict(self, inputs, labels):
        assert not self.is_training, \
            "predict(): model initialized in training mode"

        self.saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
        )) as sess:
            sess.run(tf.global_variables_initializer())
            if self.model == 1:
                # self.restore_weights(sess)
                self.saver.restore(sess, self.save_path)
                print("========Model restored from {}========".format(
                    self.save_path))
            elif self.model == 2:
                self.restore_weights(sess)
                print("--------Model restored from {}========".format(self.pickle_path))

            state = np.zeros((self.num_layers, 2, inputs.shape[1], self.num_units))
            total_loss, logits = sess.run(
                [self.loss, self.logits],
                feed_dict={self.inputs: inputs,
                           self.labels: labels,
                           self.initial_state: state}
            )

            print("\tpredict loss: {:.5f}".format(total_loss / labels.size))
            return total_loss // labels.size, logits

    def export(self):
        assert not self.is_training, \
            "export(): model initialized in training mode"

        self.saver = tf.train.Saver()

        # Define input/output names
        input_node_names = ["input_placeholder", "cudnn_lstm/state_placeholder"]
        output_node_names = ["predictions", "cudnn_lstm/output_state"]
        output_node_names_str = ",".join(output_node_names)

        # Hard-coded path values
        dir = "exports"
        graph_name = "v2"
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
