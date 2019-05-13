import numpy as np
import os
import pickle as pkl
from tqdm import tqdm
from itertools import chain
from transforms3d import quaternions as qt, euler

BLACK_LIST = [0, 1, 2, 7]
DATA_DIR = "../data/mocap"
SAVE_DIR = "processed_data"
EXPORT_DIR = "exports"
CURRENT_FPS = 120
DESIRED_FPS = 50
NUM_AUGMENTS = 7
NUM_SETS = 18


class DataStorage:

    def __init__(self, mocap_list, world_list_list, file_mapping):
        self.mocap_list = mocap_list
        self.world_list_list = world_list_list
        self.file_mapping = file_mapping
        assert len(mocap_list) == len(world_list_list)
        total_sequences = sum([data.shape[1] for data in chain.from_iterable(world_list_list)])
        self.indices = np.zeros(total_sequences, dtype="int32, int32, int32")

        idx = 0
        for i in range(len(world_list_list)):
            for j in range(len(world_list_list[i])):
                num_sequences = world_list_list[i][j].shape[1]
                for k in range(num_sequences):
                    self.indices[idx] = (i, j, k)
                    idx += 1

    def get_slice(self, index):
        """
        Gets slice at the corresponding index tuple.
        :param index: (dataset, augment, sequence)
        :return: input, label
        """

        i, j, k = index
        input = self.world_list_list[i][j][:, k, :]
        label = self.mocap_list[i][:, k, :]
        return input, label


def load_world_file(file):
    """
    Reads a world coordinate data file and returns raw data.
    :param file: file object
    :return: array [num_samples, num_channels]
    """

    num_frames = int(file.readline().split()[1])
    file.readline()  # Skip frame time
    file.readline()  # Skip joints
    bookmark = file.tell()
    num_channels = len(file.readline().split())
    file.seek(bookmark)

    frames = np.zeros((num_frames, num_channels))
    for idx in range(num_frames):
        frames[idx] = np.array([float(token) for token in file.readline().split()])

    return frames


def load_mocap_file(file):
    """
    Reads a mocap data file and returns raw data.
    :param file: file object
    :return: array [num_samples, num_channels]
    """

    while file.readline().strip() != "MOTION":
        pass
    num_frames = int(file.readline().split()[1])
    file.readline()  # Skip frame time
    bookmark = file.tell()
    num_channels = len(file.readline().split())
    file.seek(bookmark)

    frames = np.zeros((num_frames, num_channels))
    for idx in range(num_frames):
        frames[idx] = np.array([float(token) for token in file.readline().split()])

    return frames


def lerp_position(x, y, alpha):
    """
    Lerp between x and y, parameterized by alpha
    :param x: float or array of such
    :param y: float or array of such
    :param alpha: float [0-1]
    :return: float or array
    """

    return x + (y - x) * alpha


def lerp_angle(x, y, alpha):
    """
    Lerp between angles x and y, parameterized by alpha
    :param x: angle (degrees) or array of such
    :param y: angle (degrees) or array of such
    :param alpha: float [0-1]
    :return: float or array
    """

    shortest_angle = ((y-x) + 180) % 360 - 180
    return (x + alpha * shortest_angle) % 360


def resample_world_data(world_data, current_fps, desired_fps):
    """
    Re-samples world data.
    Performs the following:
        1. Up-samples to the LCM
        2. Down-samples via skipping.
    :param world_data: array [num_samples, num_channels]
    :param current_fps: int
    :param desired_fps: int
    :return: array [upsampled_num_samples, num_channels]
    """

    upsampled_fps = np.lcm(current_fps, desired_fps)
    subsamples = upsampled_fps // current_fps
    upsampled_num_samples = (world_data.shape[0] - 1)*subsamples + 1
    n_world_data = np.zeros((upsampled_num_samples, world_data.shape[1]))

    for obj_idx in range(world_data.shape[1] // 6):
        for subsample in range(subsamples):
            alpha = subsample / subsamples
            i = 6*obj_idx
            pos_x, pos_y = world_data[:-1, i:i+3], world_data[1:, i:i+3]
            rot_x, rot_y = world_data[:-1, i+3:i+6], world_data[1:, i+3:i+6]
            n_world_data[subsample:-1:subsamples, i:i+3] = lerp_position(pos_x, pos_y, alpha)
            n_world_data[subsample:-1:subsamples, i+3:i+6] = lerp_angle(rot_x, rot_y, alpha)
    n_world_data[-1] = world_data[-1]

    return n_world_data[::upsampled_fps//desired_fps]


def resample_mocap_data(mocap_data, current_fps, desired_fps):
    """
    Re-samples mocap data.
    Performs the following:
        1. Up-samples to the LCM
        2. Down-samples via skipping.
    :param mocap_data: array [num_samples, num_channels]
    :param current_fps: int
    :param desired_fps: int
    :return: array [upsampled_num_samples, num_channels]
    """

    upsampled_fps = np.lcm(current_fps, desired_fps)
    subsamples = upsampled_fps // current_fps
    upsampled_num_samples = (mocap_data.shape[0] - 1)*subsamples + 1
    n_mocap_data = np.zeros((upsampled_num_samples, mocap_data.shape[1]))

    pos_x, pos_y = mocap_data[:-1, :3], mocap_data[1:, :3]
    rot_x, rot_y = mocap_data[:-1, 3:], mocap_data[1:, 3:]
    for subsample in range(subsamples):
        alpha = subsample / subsamples
        n_mocap_data[subsample:-1:subsamples, :3] = lerp_position(pos_x, pos_y, alpha)
        n_mocap_data[subsample:-1:subsamples, 3:] = lerp_angle(rot_x, rot_y, alpha)
    n_mocap_data[-1] = mocap_data[-1]

    return n_mocap_data[::upsampled_fps//desired_fps]


def angles_to_trig_world_data(world_data):
    """
    Converts angles in world data to sin/cos values.
    :param world_data: array [num_samples, num_channels]
    :return: array [num_samples, expanded_num_channels]
    """

    expanded_num_channels = (world_data.shape[1] // 2) * 3
    n_world_data = np.zeros((world_data.shape[0], expanded_num_channels))

    for obj_idx in range(world_data.shape[1] // 6):
        i, j = 6 * obj_idx, 9 * obj_idx
        n_world_data[:, j:j + 3] = world_data[:, i:i + 3]
        n_world_data[:, j + 3:j + 9:2] = np.sin(np.radians(world_data[:, i + 3:i + 6]))
        n_world_data[:, j + 4:j + 9:2] = np.cos(np.radians(world_data[:, i + 3:i + 6]))

    return n_world_data


def angles_to_trig_mocap_data(mocap_data):
    """
    Converts angles in mocap data to sin/cos values.
    :param mocap_data: array [num_samples, num_channels]
    :return: array [num_samples, expanded_num_channels]
    """

    expanded_num_channels = (mocap_data.shape[1] - 3) * 2 + 3
    n_mocap_data = np.zeros((mocap_data.shape[0], expanded_num_channels))

    n_mocap_data[:, :3] = mocap_data[:, :3]
    n_mocap_data[:, 3::2] = np.sin(np.radians(mocap_data[:, 3:]))
    n_mocap_data[:, 4::2] = np.cos(np.radians(mocap_data[:, 3:]))

    return n_mocap_data


def convert_right_to_left(mocap_data):
    """
    Converts mocap movement data from right to left hand coordinates.
    The world_data will already have been converted in Unity.
    WARNING: DESTRUCTIVE FUNCTION
    :param mocap_data: array [num_samples, num_channels]
    :return: array [num_samples, num_channels]
    """

    # Invert x position channel
    mocap_data[:, 0] *= -1

    # Invert y and z rotation channels
    mocap_data[:, 3::3] *= -1  # z
    mocap_data[:, 5::3] *= -1  # y

    return mocap_data


def process_mocap_data(mocap_data, current_fps, desired_fps):
    """
    Processes a sequence of mocap data.
    :param mocap_data: array [num_samples, num_output_channels]
    :param current_fps: int
    :param desired_fps: int
    :return: array [adjusted_num_samples, adjusted_num_output_channels]
    """

    mocap_data = resample_mocap_data(mocap_data, current_fps, desired_fps)
    mocap_data = convert_right_to_left(mocap_data)
    mocap_data = angles_to_trig_mocap_data(mocap_data)
    mocap_data[:, :3] /= 100

    return mocap_data


def process_world_data(world_data, current_fps, desired_fps):
    """
    Processes a sequence of world data.
    :param world_data: array [num_samples, num_input_channels]
    :param current_fps: int
    :param desired_fps: int
    :return: array [adjusted_num_samples, adjusted_num_input_channels]
    """

    world_data = resample_world_data(world_data, current_fps, desired_fps)
    world_data[:, 18:] = convert_right_to_left(world_data[:, 18:])
    world_data = angles_to_trig_world_data(world_data)

    return world_data


def save_data(obj, id):
    """
    Saves an object into a pickle file with the provided id.
    :param obj: serializable obj
    :param id: string
    :return: None
    """

    file_path = SAVE_DIR + "/data-" + id + ".pkl"
    pkl.dump(obj, open(file_path, "wb"))


def process_all_data_and_save():
    counter = 0
    for file_name in tqdm(os.listdir(DATA_DIR)):
        if file_name.endswith(".bvh"):
            # Read the augment values meta file
            dir_name = file_name.replace(".bvh", "")
            augment_values_file = open(DATA_DIR + "/" + dir_name + "/augment-values.txt")
            num_augments = int(augment_values_file.readline().split()[1])

            mocap_data = load_mocap_file(open(DATA_DIR + "/" + file_name, "r"))
            mocap_data = process_mocap_data(mocap_data, CURRENT_FPS, DESIRED_FPS)
            save_data(mocap_data, str(counter) + "-mocap")
            for i in range(num_augments):
                world_data = load_world_file(
                    open(DATA_DIR + "/" + dir_name + "/" + "world-" + str(i) + ".txt", "r"))
                world_data = process_world_data(world_data, CURRENT_FPS, DESIRED_FPS)

                save_data(world_data, str(counter) + "-" + str(i))

                # # Move hips data over (result of augmentation)
                # mocap_data[:, :3] = world_data[:, 18:21]
                # mocap_data[:, 4] = world_data[:, 23]
                # mocap_data[:, 5] = world_data[:, 21]
                # mocap_data[:, 6] = world_data[:, 22]
                # world_data = world_data[:, :18]

            counter += 1


def is_valid_world_data(file_name):
    tokens = file_name.split("-")
    return int(tokens[1]) not in BLACK_LIST and "mocap" not in file_name


def sequence_split_data(data, time_len):
    """
    Splits data into sequences.
    :param data: array [num_samples, num_channels]
    :param time_len: int
    :return: array [time_len, num_sequences, num_channels]
    """

    num_sequences = data.shape[0] // time_len
    num_samples = num_sequences * time_len
    data = data[:num_samples]
    data = data.reshape((num_sequences, time_len, data.shape[1]))
    data = np.swapaxes(data, 0, 1)
    return data


# def prepare_data(time_len):
#     """
#     Retrieves all preprocessed data for model use.
#     :return: inputs, labels, inputs_valid, labels_valid
#     """
#
#     xs, ys = [], []
#     for file_name in os.listdir(SAVE_DIR):
#         if is_valid_world_data(file_name):
#             print("Skipping " + file_name)
#             continue
#         world_data, mocap_data = load_data(open(SAVE_DIR + "/" + file_name, "rb"))
#         xs.append(sequence_split_data(world_data, time_len))
#         ys.append(sequence_split_data(mocap_data, time_len))
#     x = np.concatenate(xs, axis=1)
#     y = np.concatenate(ys, axis=1)
#
#     indices = np.arange(x.shape[1])
#     np.random.shuffle(indices)
#     x = x[:, indices, :]
#     y = y[:, indices, :]
#
#     # Split into train and valid
#     valid_size = x.shape[1] // 10
#     return x[:, valid_size:], y[:, valid_size:], x[:, :valid_size], y[:, :valid_size]


def prepare_all_data(time_len):
    """
    Retrieves all preprocessed data for model use.
    :param time_len: int
    :return: DataStorage obj
    """

    mocap_list = []
    world_list_list = []
    file_mapping = []

    counter = 0
    for set_idx in range(NUM_SETS):
        if set_idx in BLACK_LIST:
            continue
        world_list_list.append([])
        mocap_data = pkl.load(open(SAVE_DIR + "/" + "data-" + str(set_idx) + "-mocap.pkl", "rb"))
        mocap_data = sequence_split_data(mocap_data, time_len)
        for aug_idx in range(NUM_AUGMENTS):
            world_data = pkl.load(
                open(SAVE_DIR + "/" + "data-" + str(set_idx) + "-" + str(aug_idx) + ".pkl", "rb"))
            world_data = sequence_split_data(world_data, time_len)
            world_list_list[counter].append(world_data)
        mocap_list.append(mocap_data)
        file_mapping.append(set_idx)
        counter += 1

    return DataStorage(mocap_list, world_list_list, file_mapping)


def prepare_indexed_data(time_len, set_idx, aug_idx):
    """
    Retrieves a particular preprocessed data for model use.
    :param time_len: int
    :param set_idx: int
    :param aug_idx: int
    :return: inputs, labels
    """

    # Load data pair
    mocap_data = pkl.load(open(SAVE_DIR + "/" + "data-" + str(set_idx) + "-mocap.pkl", "rb"))
    world_data = pkl.load(
        open(SAVE_DIR + "/" + "data-" + str(set_idx) + "-" + str(aug_idx) + ".pkl", "rb"))

    if time_len is None:
        time_len = mocap_data.shape[0]
    mocap_data = sequence_split_data(mocap_data, time_len)
    world_data = sequence_split_data(world_data, time_len)
    mocap_data, world_data = insert_root_motion(mocap_data, world_data)

    return world_data, mocap_data


def insert_root_motion(mocap_data, world_data):
    """
    Transfers root motion data from world_data to mocap_data.
    :param mocap_data: [time_len, num_sequences, output_size]
    :param world_data: [time_len, num_sequences, input_size]
    :return: mocap_data, world_data
    """

    mocap_data[:, :, :3] = world_data[:, :, 27:30]
    mocap_data[:, :, 3:5] = world_data[:, :, 34:36]
    mocap_data[:, :, 5:7] = world_data[:, :, 30:32]
    mocap_data[:, :, 7:9] = world_data[:, :, 32:34]
    world_data = world_data[:, :, :27]

    return mocap_data, world_data


def convert_logits_to_predicts(logits):
    """
    Converts logits (sin/cos) back into BVH formatted data.
    :param logits: array [time_len, 1, output_size]
    :return: array [time_len, reduced_output_size]
    """

    logits = logits[:, 0, :]
    width = (logits.shape[1] - 3) // 2 + 3
    predicts = np.zeros((logits.shape[0], width))

    # Switch from meters to centimeters
    predicts[:, :3] = logits[:, :3] * 100
    # Switch from sin/cos to angles (degrees)
    predicts[:, 3:] = np.degrees(np.arctan2(logits[:, 3::2], logits[:, 4::2]))
    return predicts


def save_predicts_to_file(predicts, pkl_index):
    """
    Saves predicts to a file.
    :param predicts: array [time_len, output_size]
    :param file_name: str
    :param pkl_index: int
    :return: None
    """

    f = open(EXPORT_DIR + "/predicts-" + str(pkl_index) + ".bvh", "w")
    for i in range(predicts.shape[0]):
        s = " ".join([str(num) for num in predicts[i]])
        f.write(s + "\n")
    print("======Predictions saved to", f.name, "======")
    f.close()


def augment_xz(inputs, labels):
    """
    Augments the data with perturbations along the x/z axis.
    Takes advantage of numpy broadcast to be fast.
    :param inputs: array [time_len, num_sequences, input_size]
    :param labels: array [time_len, num_sequences, output_size]
    :return: inputs, labels
    """

    num_sequences = inputs.shape[1]
    perturbs = np.random.uniform(-1, 1, (num_sequences, 2))

    # [time_len, num_sequences, num_obj] += [num_sequences, 1]
    inputs[:, :, ::9] += perturbs[:, 0:1]
    inputs[:, :, 1::9] += perturbs[:, 1:2]

    # [time_len, num_sequences, 1] += [num_sequences, 1]
    labels[:, :, 0:1] += perturbs[:, 0:1]
    labels[:, :, 1:2] += perturbs[:, 1:2]

    return inputs, labels


if __name__ == "__main__":
    # process_all_data_and_save()
    # import sys
    # sys.exit(0)

    # m = open(DATA_DIR + "/Take 2019-04-16 04.19.31 PM.bvh", "r")
    # w = open(DATA_DIR + "/Take 2019-04-16 04.19.31 PM/world-1.txt", "r")
    m = open(DATA_DIR + "/Take 2019-04-18 05.32.45 PM.bvh", "r")
    w = open(DATA_DIR + "/Take 2019-04-18 05.32.45 PM/world-0.txt", "r")
    # m = open(DATA_DIR + "/Take 2019-04-18 05.29.41 PM.bvh", "r")
    # w = open(DATA_DIR + "/Take 2019-04-18 05.29.41 PM/world-0.txt", "r")

    mocap_data = load_mocap_file(m)
    world_data = load_world_file(w)
    # mocap_data = convert_right_to_left(mocap_data)

    mocap_data[:, :3] = world_data[:, 18:21] * 100

    # for i in range(mocap_data.shape[0]):
    #     data = np.deg2rad(world_data[i, 21:24])
    #     q = euler.euler2quat(data[1], data[0], data[2], "syxz")
    #     e = np.rad2deg(euler.quat2euler(q, "szxy"))
    #     mocap_data[i, 3:6] = e

    mocap_data[:, 3] = world_data[:, 23]
    mocap_data[:, 4] = world_data[:, 21]
    mocap_data[:, 5] = world_data[:, 22]

    mocap_data = convert_right_to_left(mocap_data)

    # from transforms3d import quaternions as qt, euler
    # r_quat = qt.axangle2quat((0, 1, 0), np.deg2rad(-angle))
    #
    # for i in range(mocap_data.shape[0]):
    #     org_t = mocap_data[i, :3]
    #     org_r = np.deg2rad(mocap_data[i, 3:])
    #     org_r_z = qt.axangle2quat((0, 0, 1), org_r[0])
    #     org_r_x = qt.axangle2quat((1, 0, 0), org_r[1])
    #     org_r_y = qt.axangle2quat((0, 1, 0), org_r[2])
    #     new_quat = qt.qmult(r_quat, qt.qmult(org_r_y, qt.qmult(org_r_x, org_r_z)))
    #     mocap_data[i, :3] = qt.rotate_vector(org_t, r_quat)
    #     mocap_data[i, 3:6] = np.rad2deg(euler.quat2euler(new_quat, "szxy"))

    f = open(DATA_DIR + "/test-file-2.bvh", "w")
    for i in range(mocap_data.shape[0]):
        s = " ".join([str(j) for j in mocap_data[i]])
        f.write(s + "\n")
    m.close()
    f.close()

