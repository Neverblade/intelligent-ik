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
import preprocessing


class CudnnLSTMModel:
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

        #expanded_output_size = (self.output_size - 3) * 2 + 3
        # [time_len, batch_size, expanded_output_size]
        self.logits = tf.layers.dense(self.outputs, self.output_size, name="logits")

        # [time_len, batch_size, output_size-3]
        # angles = tf.stack([tf.atan2(self.logits[:, :, 2*idx+3],
        #                             self.logits[:, :, 2*idx+4])
        #                   for idx in range(0, self.output_size-3)], axis=2)
        # [time_len, batch_size, output_size]
        #self.predictions = tf.concat([self.logits[:, :, :3], angles], axis=2, name="predictions")
        # [time_len, batch_size, output_size]
        # self.difference = self.predictions - self.labels
        # float
        # self.loss = tf.reduce_mean(tf.abs(tf.atan2(tf.sin(self.difference), tf.cos(self.difference))), name="loss")

        self.loss = tf.losses.mean_squared_error(self.labels, self.logits,
                                                 reduction=tf.losses.Reduction.SUM)

        #diff = self.labels - self.logits
        #self.loss = tf.reduce_sum(tf.multiply(diff, diff))

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
                if os.path.isfile(self.save_path + ".index"):
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
            n_train = labels_.shape[1]
            indices = np.arange(n_train)
            for epoch in range(num_epochs):
                print("Epoch {}:".format(epoch))
                np.random.shuffle(indices)
                total_train_loss = 0
                num_batches = n_train // batch_size + (1 if batch_size % n_train > 0 else 0)
                for batch in tqdm(range(num_batches)):
                    current = batch * batch_size
                    _, loss = sess.run(
                        [self.train_op, self.loss],
                        feed_dict={
                            self.inputs:
                                inputs_[:, indices[current:current+batch_size], :],
                            self.labels:
                                labels_[:, indices[current:current+batch_size], :]
                        }
                    )
                    total_train_loss += loss

                # monitor per epoch
                train_loss_ = total_train_loss / labels_.size
                valid_loss_ = sess.run(
                    self.loss, feed_dict={self.inputs: inputs_valid_,
                                          self.labels: labels_valid_}) / labels_valid_.size
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
            sess.run(tf.global_variables_initializer())
            if self.model == 1:
                #self.restore_weights(sess)
                self.saver.restore(sess, self.save_path)
                print("========Model restored from {}========".format(
                    self.save_path))
            elif self.model == 2:
                self.restore_weights(sess)
                print("--------Model restored from {}========".format(self.pickle_path))

            state = np.zeros((self.num_layers, 2, inputs_test_.shape[1], self.num_units))
            total_loss = sess.run(
                self.loss,
                feed_dict={self.inputs: inputs_test_,
                           self.labels: labels_test_,
                           self.initial_state: state}
            )
            test_loss_ = total_loss / labels_test_.size
            print("\teval loss: {:.5f}".format(test_loss_))

        return test_loss_

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
            predicts = preprocessing.convert_to_predicts(logits)
            preprocessing.save_predicts(predicts, "predicts.bvh")

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
