"""
NOTES:
  To get to within 10 degrees error, error needs to be ~0.08.
  With random data fed in as input [0-5], mean error per epoch is ~0.58.
  With random inputs [0-5] and random outputs [0-6], mean error is ~1.6
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import getopt
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

dir_path = "./checkpoints"

# Command-line Params
plot_error = False                                                  # -p --plot_error
testing = False                                                     # -r --testing
use_checkpoint = False                                              # -c --use_checkpoint
sequence_length = 512                                               # -s --sequence_length
training_batch_size = 128                                           # -b --batch_size
truncated_backprop_length = 16                                      # -t --truncated_backprop_length
learning_rate = 0.001                                               # -l --learning_rate
data_files = ["toy/data1.txt", "toy/data2.txt", "toy/data3.txt"]    # -d --data_files

# Training Params
num_epochs = 100000
num_validation_samples = 10000

# Architecture Params
units_per_layer = [16]  # Will also include a final RNN layer of output size
dropout_per_layer = [1]


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


def load_data_basic():
    print("Loading data...")

    num_sequences_per_file = [get_num_samples(file_name) // sequence_length for file_name in data_files]
    total_num_sequences = sum(num_sequences_per_file)
    total_num_samples = total_num_sequences * sequence_length

    raw_x = np.zeros((total_num_samples, 4))
    raw_y = np.zeros((total_num_samples, 2))
    sample_counter = 0

    for file_idx in range(len(data_files)):
        file_name = data_files[file_idx]
        num_sequences = num_sequences_per_file[file_idx]
        f = open("data/" + file_name, "r")
        f.readline()  # Skip # of frames
        f.readline()  # Skip frame time
        for _ in range(num_sequences * sequence_length):
            if sample_counter >= total_num_samples:
                break

            tokens = f.readline().split()
            angle1, angle2 = np.radians(float(tokens[2])), np.radians(float(tokens[3]))
            angle = (angle1 + angle2) % (2 * math.pi)
            sin_a, cos_a = np.sin(angle), np.cos(angle)

            raw_x[sample_counter][0] = float(tokens[0])
            raw_x[sample_counter][1] = float(tokens[1])
            raw_x[sample_counter][2] = sin_a
            raw_x[sample_counter][3] = cos_a

            raw_y[sample_counter][0] = angle1
            raw_y[sample_counter][1] = angle2

            sample_counter += 1

    x = raw_x.reshape((total_num_sequences, sequence_length, 4))
    y = raw_y.reshape((total_num_sequences, sequence_length, 2))
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
        new_y[seq_idx, :, 2:4] = np.mod(np.negative(y[2 * seq_idx + 1]), 2 * math.pi)

        # Add the label angles to get the angle of the input point
        summed_angles = np.mod(np.sum(new_y[seq_idx, :, 2:4], axis=1), 2 * math.pi)
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
    lower_bound, upper_bound = -math.pi / 2, 3 * math.pi / 2
    t = t * (upper_bound - lower_bound) + lower_bound

    # Sample
    return np.array([math.cos(t), math.sin(t) * math.cos(t)])


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


##################
# DEFINING GRAPH #
##################

"""
Builds the RNN graph.
Args:
  X: input data
  Y: labels
Returns:
  Dictionary of tf nodes
"""


def build_graph(x, y):
    print("Building graph...")

    # Inputs
    batch_x_placeholder = tf.placeholder(tf.float32, [None, None, x.shape[2]], name="input_X")
    batch_y_placeholder = tf.placeholder(tf.float32, [None, None, y.shape[2]], name="input_Y")

    # Defining cells and RNN
    state_placeholders = tuple(
        tf.placeholder(tf.float32, [None, units_per_layer[idx]], name="input_state_" + str(idx)) for idx in
        range(len(units_per_layer)))
    cells = [tf.contrib.rnn.DropoutWrapper(
        tf.contrib.rnn.BasicRNNCell(num_units=units_per_layer[idx]),
        output_keep_prob=dropout_per_layer[idx])
        for idx in range(len(units_per_layer))]
    cell = tf.contrib.rnn.MultiRNNCell(cells)

    # RNN layers
    output, state = tf.nn.dynamic_rnn(cell, batch_x_placeholder, initial_state=state_placeholders)
    output_states = tuple(tf.identity(state[idx], name="output_state_" + str(idx)) for idx in range(len(state)))

    # Convert final output to angles
    predictions = tf.stack([tf.atan2(output[:, :, 2 * idx], output[:, :, 2 * idx + 1]) for idx in range(y.shape[2])],
                           axis=2, name="output")

    # Compute error
    diff = batch_y_placeholder - predictions
    total_loss = tf.reduce_mean(tf.abs(tf.atan2(tf.sin(diff), tf.cos(diff))))

    # Train
    train_step = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08
    ).minimize(total_loss)

    return {
        "batchX_placeholder": batch_x_placeholder,
        "batchY_placeholder": batch_y_placeholder,
        "state_placeholders": state_placeholders,
        "total_loss": total_loss,
        "train_step": train_step,
        "state": state
    }


##################
# EPOCH FUNCTION #
##################

"""
Trains on data X and labels Y.
Iterates once through the data.
Returns the average error across all iterations.
"""


def perform_epoch(sess, nodes, x, y, batch_size, evaluating=False):
    num_sequences = x.shape[0]
    num_batches = num_sequences // batch_size
    indices = np.arange(num_sequences)
    np.random.shuffle(indices)
    sum_of_losses, num_losses = 0, 0

    # Detect if there aren't enough batches
    if num_batches == 0:
        print("  Parameters resulted in 0 batch iterations.")
        print("  num_sequences:", num_sequences, "batch_size:", batch_size)
        print("  Aborting.")
        sys.exit(0)

    for batch_idx in range(num_batches):
        # Reset state
        _state = tuple(np.zeros((batch_size, n)) for n in units_per_layer)

        # Load in sequences
        batch_sequence_x = np.zeros((batch_size, sequence_length, x.shape[2]))
        batch_sequence_y = np.zeros((batch_size, sequence_length, y.shape[2]))
        for seq_offset in range(batch_size):
            seq_idx = indices[batch_size * batch_idx + seq_offset]
            batch_sequence_x[seq_offset] = x[seq_idx]
            batch_sequence_y[seq_offset] = y[seq_idx]

        # Iterate through sequence
        sample_idx = 0
        while sample_idx < sequence_length:
            batch_x = batch_sequence_x[:, sample_idx:sample_idx + truncated_backprop_length, :]
            batch_y = batch_sequence_y[:, sample_idx:sample_idx + truncated_backprop_length, :]
            feed_dict = {
                nodes["batchX_placeholder"]: batch_x,
                nodes["batchY_placeholder"]: batch_y,
                nodes["state_placeholders"]: _state
            }

            if not evaluating:
                _total_loss, _train_step, _state = sess.run([nodes["total_loss"], nodes["train_step"], nodes["state"]],
                                                            feed_dict)
            else:
                _total_loss, _state = sess.run([nodes["total_loss"], nodes["state"]], feed_dict)

            sum_of_losses += _total_loss
            num_losses += 1
            sample_idx += truncated_backprop_length

    return sum_of_losses / num_losses


def plot_loss(fig, ax, training_losses, validation_losses):
    ax.clear()
    x_axes = np.arange(len(training_losses))
    ax.plot(x_axes, training_losses, color="blue")
    ax.plot(x_axes, validation_losses, color="red")
    fig.canvas.draw()


###########
# GENERAL #
###########

def process_args():
    print("Processing arguments...")
    options = "prcs:b:t:l:d:"
    long_options = [
        "plot_error",
        "testing",
        "use_checkpoint",
        "sequence_length=",
        "batch_size=",
        "truncated_backprop_length=",
        "learning_rate=",
        "data_files="]
    try:
        opts, args = getopt.getopt(sys.argv[1:], options, long_options)
    except getopt.GetoptError:
        print(
            "toy_training.py -p -c -r " +
            "-s <sequence_length> -b <batch_size> -t <truncated_backprop_length> -l <learning_rate> -d <data_files>")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-p", "--plot_error"):
            global plot_error
            plot_error = True
        elif opt in ("-r", "--testing"):
            global testing
            testing = True
        elif opt in ("-c", "--use_checkpoint"):
            global use_checkpoint
            use_checkpoint = True
        elif opt in ("-s", "--sequence_length"):
            global sequence_length
            sequence_length = int(arg)
        elif opt in ("-b", "--batch_size"):
            global batch_size
            batch_size = int(arg)
        elif opt in ("-t", "--truncated_backprop_length"):
            global truncated_backprop_length
            truncated_backprop_length = int(arg)
        elif opt in ("-l", "--learning_rate"):
            global learning_rate
            learning_rate = float(arg)
        elif opt in ("-d", "--data_files"):
            global data_files
            data_files = arg.split()


def export_model(sess):
    print("Saving graph...")

    saver = tf.train.Saver()
    input_node_names = ["input_state_" + str(idx) for idx in range(len(units_per_layer))]
    input_node_names.append("input_X")
    output_node_names = ["output_state_" + str(idx) for idx in range(len(units_per_layer))]
    output_node_names.append("output")
    output_node_names_str = ",".join(output_node_names)

    DIR = "exports"
    GRAPH_NAME = "toy"
    pbtxt = DIR + "/" + GRAPH_NAME + "_graph.pbtxt"
    chkp = DIR + "/" + GRAPH_NAME + ".chkp"

    tf.train.write_graph(sess.graph_def, DIR, GRAPH_NAME + "_graph.pbtxt")
    saver.save(sess, chkp)
    freeze_graph.freeze_graph(pbtxt,  # input_graph
                              None,  # input_saver
                              False,  # input_binary
                              chkp,  # input_checkpoint
                              output_node_names_str,  # output_node_names
                              "save/restore_all",
                              "save/Const:0",
                              DIR + "/" + "frozen_" + GRAPH_NAME + ".bytes",
                              True,
                              "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(DIR + "/frozen_" + GRAPH_NAME + ".bytes", "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def, input_node_names, output_node_names,
        tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile(DIR + "/opt_" + GRAPH_NAME + ".bytes", "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("Graph saved.")


def main():
    process_args()

    # Custom settings for test time
    if testing:
        global sequence_length
        global truncated_backprop_length
        global training_batch_size
        sequence_length = sum([get_num_samples(file_name) for file_name in data_files])
        truncated_backprop_length = 1
        training_batch_size = 1

    # Load data and transform it
    x, y = load_data_basic()
    # X, Y = transform_root_position(X, Y)

    # Update layers with the output. Final layer is 2x because theta -> sin(theta) cos(theta)
    units_per_layer.append(2 * y.shape[2])
    dropout_per_layer.append(1.0)

    # Create the graph and saver
    nodes = build_graph(x, y)
    saver = tf.train.Saver(max_to_keep=4)

    with tf.Session() as sess:
        # Reset weights
        sess.run(tf.global_variables_initializer())

        # Load a model if specified. This should be set during testing.
        if use_checkpoint:
            print("Restoring checkpoint...")
            saver.restore(sess, tf.train.latest_checkpoint(dir_path))

        # EXPORT MODEL TO UNITY
        export_model(sess)
        sys.exit(0)

        if not testing:
            print("Starting training...")

            # Define training and validation sets
            num_validation_sequences = num_validation_samples // sequence_length
            training_x = x[num_validation_sequences:]
            training_y = y[num_validation_sequences:]
            validation_x = x[:num_validation_sequences]
            validation_y = y[:num_validation_sequences]

            # Initialize plot
            if plot_error:
                plt.ion()
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)

            # Training loop
            training_losses, validation_losses = [], []
            for epoch_idx in range(num_epochs):
                training_loss = perform_epoch(
                    sess, nodes, training_x, training_y, training_batch_size)
                training_losses.append(training_loss)
                validation_loss = perform_epoch(
                    sess, nodes, validation_x, validation_y, 8, evaluating=True)
                validation_losses.append(validation_loss)
                print("Epoch:",
                      "{:<4}".format(epoch_idx),
                      "T-Loss:", training_loss,
                      "V-Loss:", validation_loss)
                if plot_error:
                    plot_loss(fig, ax, training_losses, validation_losses)
                if epoch_idx % 50 == 0 and epoch_idx != 0:
                    saver.save(sess, dir_path + "/toy", global_step=epoch_idx, write_meta_graph=False)

            if plot_error:
                plt.close(fig)
        else:
            print("Starting testing...")
            testing_loss = perform_epoch(sess, nodes, x, y, training_batch_size, evaluating=True)
            print("Testing Loss:", testing_loss)


if __name__ == "__main__":
    main()
