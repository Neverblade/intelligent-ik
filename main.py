"""
Random input gets errors of ~0.25-0.35.
"""

import tensorflow as tf
from cudnnlstm import CudnnLSTMModel
import preprocessing
from preprocessing import VRData, MocapData
import numpy as np

# allow global hyperparameters using `tf.app.flags`
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("time_len", 32,
                            """Number of timesteps per sequence.""")
tf.app.flags.DEFINE_integer("input_size", 27,
                            """Dimension of inputs.""")
tf.app.flags.DEFINE_integer("output_size", 81,
                            """Dimension of outputs.""")

tf.app.flags.DEFINE_integer("num_layers", 2,
                            """Number of stacked LSTM layers.""")
tf.app.flags.DEFINE_integer("num_units", 64,
                            """Number of units in an LSTM layer.""")
tf.app.flags.DEFINE_string("direction", "unidirectional",
                           """Direction of the LSTM RNN. 
                              Either `unidirectional` or `bidirectional`.""")

tf.app.flags.DEFINE_integer("num_epochs", 1000,
                            """Number of epochs for training.""")
tf.app.flags.DEFINE_integer("batch_size", 256,
                            """Batch size per iteration during training.""")
tf.app.flags.DEFINE_float("learning_rate", 0.001,
                          """Learning rate for training using Adam.""")
tf.app.flags.DEFINE_float("dropout", 0.2,
                          """Dropout probability. `0.` means no dropout.""")

tf.app.flags.DEFINE_integer("seed", 0,
                            """Random seed for both numpy and tensorflow.""")

tf.app.flags.DEFINE_integer("model", 1,
                            """CudnnLSTM (0), LSTMBlockCell (1), or LSTMCell (2).""")
tf.app.flags.DEFINE_integer("action", 3,
                            """Whether to train (0), eval (1), export (2), or predict (3).""")


def main(_):

    if FLAGS.seed != -1:
        np.random.seed(FLAGS.seed)
        tf.set_random_seed(FLAGS.seed)

    # Random data
    # sequences = 1000
    # x = np.random.uniform(-1, 1, (FLAGS.time_len, sequences, FLAGS.input_size))
    # y = np.random.uniform(-1, 1, (FLAGS.time_len, sequences, FLAGS.output_size))
    # valid_size = x.shape[1] // 10
    # inputs, labels, inputs_valid, labels_valid \
    #     = x[:, valid_size:], y[:, valid_size:], x[:, :valid_size], y[:, :valid_size]

    print("Creating model...")
    model = CudnnLSTMModel(FLAGS.input_size, FLAGS.output_size,
                           FLAGS.num_layers, FLAGS.num_units, FLAGS.direction,
                           FLAGS.learning_rate, FLAGS.dropout, FLAGS.seed,
                           is_training=FLAGS.action == 0,
                           model=FLAGS.model)

    if FLAGS.action == 0:  # TRAINING
        #assert FLAGS.model == 0 or FLAGS.model == 2, \
        #    "main(): trained model must be CudnnLSTM, or LSTMCell"
        inputs, labels, inputs_valid, labels_valid = preprocessing.get_data(FLAGS.time_len)
        model.train(inputs, inputs_valid, labels, labels_valid,
                    FLAGS.batch_size, FLAGS.num_epochs)
    elif FLAGS.action == 1:  # EVALUATING
        assert FLAGS.model == 1 or FLAGS.model == 2, \
            "main(): evaluated model must be LSTMBlockCell or LSTMCell"
        inputs, labels, inputs_valid, labels_valid = preprocessing.get_data(FLAGS.time_len)
        model.eval(inputs_valid, labels_valid)
    elif FLAGS.action == 2:  # EXPORTING
        assert FLAGS.model == 1 or FLAGS.model == 2, \
            "main(): evaluated model must be LSTMBlockCell or LSTMCell"
        if FLAGS.model == 1:
            model.export_weights()
        elif FLAGS.model == 2:
            model.export()
    elif FLAGS.action == 3:  # PREDICTING
        assert FLAGS.model == 1 or FLAGS.model == 2, \
            "main(): predicting model must be LSTMBlockCell or LSTMCell"
        mocap_data, vr_data = preprocessing.load_data("data-4.txt")
        inputs = preprocessing.shape_data(vr_data.frames, vr_data.frames.shape[0])
        labels = preprocessing.shape_data(mocap_data.frames, mocap_data.frames.shape[0])[:, :, :81]
        model.predict(inputs, labels)

        # inputs, labels, inputs_valid, labels_valid = preprocessing.get_data(FLAGS.time_len)
        # print(inputs_valid.shape, labels_valid.shape)
        # np.swapaxes(inputs_valid, 0, 1)
        # inputs_valid = inputs_valid.reshape((inputs_valid.shape[0] * inputs_valid.shape[1], inputs_valid.shape[2]))
        # np.swapaxes(labels_valid, 0, 1)
        # labels_valid = labels_valid.reshape((labels_valid.shape[0] * labels_valid.shape[1], labels_valid.shape[2]))
        # print(inputs_valid.shape, labels_valid.shape)
        # inputs = preprocessing.shape_data(inputs_valid, inputs_valid.shape[0])
        # labels = preprocessing.shape_data(labels_valid, labels_valid.shape[0])
        # print(inputs.shape, labels.shape)
        # model.predict(inputs, labels)


if __name__ == "__main__":
    tf.app.run(main=main)
