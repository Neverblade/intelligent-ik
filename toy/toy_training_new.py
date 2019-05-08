"""
NOTES:
  To get to within 10 degrees error, error needs to be ~0.08.
  With random data fed in as input [0-5], mean error per epoch is ~0.58.
  With random inputs [0-5] and random outputs [0-6], mean error is ~1.6
"""

import tensorflow as tf
import numpy as np
from gru import GRUToyModel

dir_path = "./checkpoints"
data_files = ["data1.txt", "data2.txt", "data3.txt"]    # -d --data_files

# allow global hyperparameters using `tf.app.flags`
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("n", 10000,
                            """Number of synthetic data to be used.""")
tf.app.flags.DEFINE_integer("time_len", 512,
                            """Number of timesteps in synthetic data.""")
tf.app.flags.DEFINE_integer("input_size", 4,
                            """Dimension of inputs. Fix to 2.""")

tf.app.flags.DEFINE_integer("num_layers", 1,
                            """Number of stacked LSTM layers.""")
tf.app.flags.DEFINE_integer("num_units", 16,
                            """Number of units in an LSTM layer.""")
tf.app.flags.DEFINE_string("direction", "unidirectional",
                           """Direction of the LSTM RNN. 
                              Either `unidirectional` or `bidirectional`.""")

tf.app.flags.DEFINE_integer("num_epochs", 500,
                            """Number of epochs for training.""")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            """Batch size per iteration during training.""")
tf.app.flags.DEFINE_float("learning_rate", 0.001,
                          """Learning rate for training using Adam.""")
tf.app.flags.DEFINE_float("dropout", 0.2,
                          """Dropout probability. `0.` means no dropout.""")

tf.app.flags.DEFINE_integer("seed", 0,
                            """Random seed for both numpy and tensorflow.""")

tf.app.flags.DEFINE_integer("model", 2,
                            """CudnnLSTM (0), LSTMBlockCell (1), or LSTMCell (2).""")
tf.app.flags.DEFINE_integer("action", 1,
                            """Whether to train (0), eval (1), or export (2).""")

#################
# READING INPUT #
#################


# Returns the # of samples in a file
def get_num_samples(file_name):
    f = open("data/" + file_name, "r")
    n = int(f.readline().split()[1])
    f.close()
    return n


"""
Returns:
  Inputs: x, y, sin(theta), cos(theta)
  Outputs: theta1, theta2
"""


def load_data_basic(time_len):
    print("Loading data...")

    num_sequences_per_file = [get_num_samples(file_name) // time_len for file_name in data_files]
    total_num_sequences = sum(num_sequences_per_file)
    total_num_samples = total_num_sequences * time_len

    raw_x = np.zeros((total_num_samples, 4))
    raw_y = np.zeros((total_num_samples, 2))
    sample_counter = 0

    for file_idx in range(len(data_files)):
        file_name = data_files[file_idx]
        num_sequences = num_sequences_per_file[file_idx]
        f = open("data/" + file_name, "r")
        f.readline()  # Skip # of frames
        f.readline()  # Skip frame time
        for _ in range(num_sequences * time_len):
            if sample_counter >= total_num_samples:
                break

            tokens = f.readline().split()
            angle1, angle2 = np.radians(float(tokens[2])), np.radians(float(tokens[3]))
            angle = (angle1 + angle2) % (2 * np.pi)
            sin_a, cos_a = np.sin(angle), np.cos(angle)

            raw_x[sample_counter][0] = float(tokens[0])
            raw_x[sample_counter][1] = float(tokens[1])
            raw_x[sample_counter][2] = sin_a
            raw_x[sample_counter][3] = cos_a

            raw_y[sample_counter][0] = angle1
            raw_y[sample_counter][1] = angle2

            sample_counter += 1

    x = raw_x.reshape((total_num_sequences, time_len, 4))
    x = np.swapaxes(x, 0, 1)
    y = raw_y.reshape((total_num_sequences, time_len, 2))
    y = np.swapaxes(y, 0, 1)
    return x, y


"""
Inputs:
  X: x, y, sin(theta), cos(theta)
  Y: angle1, angle2
Outputs:
  Doubling up on the X and Y to create 2 arms.
"""


def transform_two_arms(x, y):
    print("Transforming to two arms...")

    num_sequences = x.shape[0] // 2
    new_x = np.zeros((num_sequences, sequence_length, 8))
    new_y = np.zeros((num_sequences, sequence_length, 4))

    for seq_idx in range(num_sequences):
        # Feed in first arm as is
        new_x[seq_idx, :, :4] = x[2 * seq_idx]
        new_y[seq_idx, :, :2] = y[2 * seq_idx]

        # Flip the x/y of the second arm
        new_x[seq_idx, :, 4:6] = x[2 * seq_idx + 1, :, :2]

        # Flip the angles of the labels
        new_y[seq_idx, :, 2:4] = np.mod(np.negative(y[2 * seq_idx + 1]), 2 * np.pi)

        # Add the label angles to get the angle of the input point
        summed_angles = np.mod(np.sum(new_y[seq_idx, :, 2:4], axis=1), 2 * np.pi)
        new_x[seq_idx, :, 6] = np.sin(summed_angles)
        new_x[seq_idx, :, 7] = np.cos(summed_angles)

    return new_x, new_y


"""
Samples an "infinity symbol" periodic function
Args:
  t: parameter to function. t goes [0-1] and maps to the period
Returns:
  [x, y]
"""


def sample_lemniscate(t):
    # Loop between [-pi/2, 3pi/2]
    t %= 1
    lower_bound, upper_bound = -np.pi / 2, 3 * np.pi / 2
    t = t * (upper_bound - lower_bound) + lower_bound

    # Sample
    return np.array([np.cos(t), np.sin(t) * np.cos(t)])


"""
Inputs:
  X: x, y, sun(theta), cos(theta)
  Y: angle1, angle2
Outputs:
  Same thing, but the root moves in a smooth motion
"""


def transform_root_position(x, y):
    print("Transforming root positions...")

    period = 5  # seconds
    size = 0.5  # meters
    time_per_frame = 0.02  # seconds

    t = 0
    for seq_idx in range(x.shape[0]):
        for sample_idx in range(x.shape[1]):
            offset = sample_lemniscate(t) * size
            x[seq_idx, sample_idx, :2] += offset
            t += time_per_frame / period

    return x, y


def main(_):
    if FLAGS.action != 2:
        x, y, = load_data_basic(FLAGS.time_len)
        valid_size = 10000 // FLAGS.time_len
        inputs_, inputs_valid_ = x[:, valid_size:, :], x[:, :valid_size, :]
        labels_, labels_valid_ = y[:, valid_size:, :], y[:, :valid_size, :]

    model = GRUToyModel(FLAGS.input_size,
                     FLAGS.num_layers, FLAGS.num_units, FLAGS.direction,
                     FLAGS.learning_rate, FLAGS.dropout, FLAGS.seed,
                     is_training=FLAGS.action == 0,
                     model=FLAGS.model)

    if FLAGS.action == 0:  # TRAINING
        assert FLAGS.model == 0 or FLAGS.model == 2, \
            "main(): trained model must be CudnnLSTM, or LSTMCell"
        model.train(inputs_, inputs_valid_, labels_, labels_valid_,
                    FLAGS.batch_size, FLAGS.num_epochs)
    elif FLAGS.action == 1:  # TESTING
        assert FLAGS.model == 1 or FLAGS.model == 2, \
            "main(): evaluated model must be LSTMBlockCell or LSTMCell"
        model.eval(inputs_valid_, labels_valid_)
    elif FLAGS.action == 2:  # EXPORTING
        assert FLAGS.model == 1 or FLAGS.model == 2, \
            "main(): evaluated model must be LSTMBlockCell or LSTMCell"
        if FLAGS.model == 1:
            model.export_weights()
        elif FLAGS.model == 2:
            model.export()


if __name__ == "__main__":
    tf.app.run(main=main)