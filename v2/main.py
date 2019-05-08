import tensorflow as tf
from lstm import LSTMModel
import data_processing
import numpy as np

# allow global hyperparameters using `tf.app.flags`
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("time_len", 32,
                            """Number of timesteps per sequence.""")
tf.app.flags.DEFINE_integer("input_size", 27,
                            """Dimension of inputs.""")
tf.app.flags.DEFINE_integer("output_size", 129,
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

tf.app.flags.DEFINE_integer("model", 0,
                            """CudnnLSTM (0), LSTMBlockCell (1), or LSTMCell (2).""")
tf.app.flags.DEFINE_integer("action", 0,
                            """Whether to train (0), eval (1), export (2), or predict (3).""")


def main(_):
    if FLAGS.seed != -1:
        np.random.seed(FLAGS.seed)
        tf.set_random_seed(FLAGS.seed)

    print("Creating model...")
    model = LSTMModel(FLAGS.input_size, FLAGS.output_size,
                           FLAGS.num_layers, FLAGS.num_units, FLAGS.direction,
                           FLAGS.learning_rate, FLAGS.dropout, FLAGS.seed,
                           is_training=FLAGS.action == 0,
                           model=FLAGS.model)

    if FLAGS.action == 0:  # TRAINING
        assert FLAGS.model != 1
        data_storage = data_processing.prepare_all_data(FLAGS.time_len)
        model.train(data_storage, FLAGS.batch_size, FLAGS.num_epochs)
        # inputs, labels, inputs_valid, labels_valid = data_processing.prepare_data(FLAGS.time_len)
        # model.train(inputs, inputs_valid, labels, labels_valid,
        #             FLAGS.batch_size, FLAGS.num_epochs)
    elif FLAGS.action == 1:  # EVALUATING
        assert FLAGS.model == 1 or FLAGS.model == 2, \
            "main(): evaluated model must be LSTMBlockCell or LSTMCell"

        set_idx, aug_idx = 7, 0
        inputs, labels = data_processing.prepare_indexed_data(FLAGS.time_len, set_idx, aug_idx)
        model.predict(inputs, labels)

        # pkl_index = 7
        # world_data, mocap_data = data_processing.load_data(
        #     open("processed_data/data-" + str(pkl_index) + ".pkl", "rb"))
        # inputs = data_processing.sequence_split_data(world_data, world_data.shape[0])
        # labels = data_processing.sequence_split_data(mocap_data, mocap_data.shape[0])
        # #inputs, labels = data_processing.augment_data(inputs, labels)
        # model.predict(inputs, labels)
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

        set_idx, aug_idx = 7, 0
        inputs, labels = data_processing.prepare_indexed_data(FLAGS.time_len, set_idx, aug_idx)
        predicts = model.predict(inputs, labels)
        data_processing.save_predicts_to_file(predicts, set_idx)

        # pkl_index = 7
        # world_data, mocap_data = data_processing.load_data(
        #     open("processed_data/data-" + str(pkl_index) + ".pkl", "rb"))
        # inputs = data_processing.sequence_split_data(world_data, world_data.shape[0])
        # labels = data_processing.sequence_split_data(mocap_data, mocap_data.shape[0])
        # loss, logits = model.predict(inputs, labels)
        # predicts = data_processing.convert_logits_to_predicts(logits)
        # data_processing.save_predicts_to_file(predicts, pkl_index)


if __name__ == "__main__":
    tf.app.run(main=main)
