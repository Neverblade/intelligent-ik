"""
Contains utilities for loading and preprocessing data.
"""

import numpy as np
import pickle as pkl
from tqdm import tqdm
import os

data = [
    (
        "2019-16-4--17-32-20-processed.txt",
        "Take 2019-04-16 05.19.49 PM.bvh",
        3.816666667
    ),
    (
        "2019-18-4--16-44-36-processed.txt",
        "Take 2019-04-18 04.08.25 PM.bvh",
        10.96833333
    ),
    (
        "2019-18-4--17-32-50-processed.txt",
        "Take 2019-04-18 04.57.43 PM.bvh",
        6.168333333
    ),
    (
        "2019-18-4--17-36-33-processed.txt",
        "Take 2019-04-18 05.29.41 PM.bvh",
        -3.196666667
    ),
    (
        "2019-18-4--17-38-04-processed.txt",
        "Take 2019-04-18 05.32.45 PM.bvh",
        -3.56
    ),
    (
        "2019-23-4--14-40-45-processed.txt",
        "Take 2019-04-23 02.36.48 PM.bvh",
        9.603333333
    ),
    # (
    #     "2019-23-4--15-15-47-processed.txt",
    #     "Take 2019-04-23 03.11.37 PM.bvh",
    #     -2.87666666666666
    # ),
    (
        "2019-23-4--15-50-26-processed.txt",
        "Take 2019-04-23 03.46.23 PM.bvh",
        4.175
    ),
    (
        "2019-25-4--14-31-11-processed.txt",
        "Take 2019-04-25 02.27.06 PM.bvh",
        3.92666666666666
    ),
    (
        "2019-25-4--15-06-41-processed.txt",
        "Take 2019-04-25 03.02.37 PM.bvh",
        4.34833333333333
    ),
    (
        "2019-25-4--15-43-39-processed.txt",
        "Take 2019-04-25 03.40.13 PM.bvh",
        42.7816666666666
    ),
    (
        "2019-25-4--16-00-11-processed.txt",
        "Take 2019-04-25 03.56.06 PM.bvh",
        4.65666666666666
    )
]


class VRData:

    def __init__(self, num_frames, fps, calibration_frame, frames):
        self.num_frames = num_frames
        self.fps = fps
        self.calibration_frame = calibration_frame
        self.frames = frames


class MocapData:

    def __init__(self, num_frames, fps, frames):
        # TODO: decide if the skeleton data is required as well
        self.num_frames = num_frames
        self.fps = fps
        self.frames = frames


def load_vr_file(file):
    """
    Reads a VR data file and returns the raw data.
    :param file: string
    :return: VRData
    """

    f = open(file, "r")
    num_frames = int(f.readline().split()[1])
    fps = int(np.round(1 / float(f.readline().split()[2])))
    calibration_frame = np.array([float(token) for token in f.readline().split()[1:]])
    num_channels = calibration_frame.size
    frames = np.zeros((num_frames, num_channels))

    for i in range(num_frames):
        frames[i] = np.array([float(token) for token in f.readline().split()])

    return VRData(num_frames, fps, calibration_frame, frames)


def load_mocap_file_helper(f):
    """
    Reads a a mocap file that's been stripped of its skeleton definition and returns the raw data.
    :param f: file object
    :return: MocapData
    """

    num_frames = int(f.readline().split()[1])
    fps = int(np.round(1 / float(f.readline().split()[2])))
    bookmark = f.tell()
    num_channels = len(f.readline().split())
    f.seek(bookmark)
    frames = np.zeros((num_frames, num_channels))

    for i in range(num_frames):
        frames[i] = np.array([float(token) for token in f.readline().split()])

    return MocapData(num_frames, fps, frames)


def load_mocap_file(file):
    """
    Reads a mocap data file and returns the raw data.
    :param file: name
    :return: MocapData
    """

    f = open(file, "r")
    while f.readline().strip() != "MOTION":
        pass
    return load_mocap_file_helper(f)


def preprocess_data(mocap_data, vr_data, offset, desired_fps):
    """
    Preprocesses mocap_data and vr_data. Performs the offset and sampling required.
    :param mocap_data: MocapData object
    :param vr_data: VRData object (processed)
    :param offset: temporal offset - move mocap_data forward in time.
    :param desired_fps: fps each sequence should end up at. assumed to divisble under lcm of fps's
    :return: None
    """

    # Up-sample both
    mocap_fps, vr_fps = mocap_data.fps, vr_data.fps
    fps = np.lcm(mocap_fps, vr_fps)
    upsample_mocap_vec(mocap_data, fps)
    upsample_vr_vec(vr_data, fps)
    mocap_vec, vr_vec = mocap_data.frames, vr_data.frames

    # Apply offset and down-sample
    sample_offset, skip = int(round(fps*offset)), fps//desired_fps
    mocap_vec = mocap_vec[max(-sample_offset, 0):]
    vr_vec = vr_vec[max(sample_offset, 0):]
    min_length = min(mocap_vec.shape[0], vr_vec.shape[0])
    mocap_vec = mocap_vec[:min_length:skip]
    vr_vec = vr_vec[:min_length:skip]
    mocap_data.num_frames, mocap_data.frames = mocap_vec.shape[0], mocap_vec
    vr_data.num_frames, vr_data.frames = vr_vec.shape[0], vr_vec
    vr_data.fps, mocap_data.fps = desired_fps, desired_fps

    # Convert angles to sin/cos
    convert_vr_vec_to_sin_cos(vr_data)
    convert_mocap_vec_to_sin_cos(mocap_data)

    # Scale down positions of mocap (centimeters to meters)
    mocap_data.frames[:, :3] /= 100


def save_data(mocap_data, vr_data, name):
    """
    Saves (hopefully preprocessed) data into a single file.
    :param mocap_data: MocapData obj
    :param vr_data: VRData obj
    :param name: name of the file
    :return: None
    """

    assert mocap_data.num_frames == vr_data.num_frames \
        and mocap_data.fps == vr_data.fps

    file_path = "../data/preprocessed/" + name + ".txt"
    save_obj = {"mocap": mocap_data, "vr": vr_data}
    pkl.dump(save_obj, open(file_path, "wb"))


def load_data(name):
    """
    Loads a preprocessed data from a file.
    :param name: name of the file
    :return: (mocap_data, vr_data)
    """

    file_path = "../data/preprocessed/" + name
    save_obj = pkl.load(open(file_path, "rb"))
    return save_obj["mocap"], save_obj["vr"]


def shape_data(vec, time_len):
    """
    Shapes data into the time-major, sequenced form required by the LSTM model
    :param vec: np array [num_samples, size]
    :param time_len: int, # of samples per sequence
    :return: np array [time_len, num_sequences, size]
    """

    num_sequences = vec.shape[0] // time_len
    num_samples = num_sequences * time_len
    vec = vec[:num_samples]
    vec = vec.reshape((num_sequences, time_len, vec.shape[1]))
    vec = np.swapaxes(vec, 0, 1)
    return vec


def transform_vr_frame(frame):
    """
    Transforms a single frame of vr input. Converts angles to sin/cos values.
    :param frame: 1D np array
    :return: 1D np array
    """

    new_frame = np.zeros(27)  # TODO: Don't hardcode this
    for i in range(3):
        f_index, nf_index = 6*i, 9*i
        new_frame[nf_index:nf_index+3] = frame[f_index:f_index+3]
        for j in range(3):
            new_frame[nf_index+3+2*j] = np.sin(frame[f_index+3+j])
            new_frame[nf_index+3+2*j+1] = np.cos(frame[f_index+3+j])
    return new_frame


def transform_mocap_frame(frame):
    """
    Transforms a single frame of mocap input. Converts angles to sin/cos values.
    :param frame: 1D np array
    :return: 1D np array
    """

    # First 3 floats are xyz. Remainder are angles.
    new_frame = np.zeros(3 + (frame.size-3) * 2)
    new_frame[:3] = frame[:3]
    for i in range(frame.size - 3):
        f_index, nf_index = i + 3, 2*i + 3
        new_frame[nf_index] = np.sin(frame[f_index])
        new_frame[nf_index+1] = np.cos(frame[f_index])
    return new_frame


def convert_vr_vec_to_sin_cos(vr_data):
    """
    Converts all angles in vr_data to sin/cos values
    :param vr_data: VRData object
    :return: None
    """

    vr_vec = vr_data.frames
    n_vr_vec = np.zeros((vr_vec.shape[0], 27))  # 9 (pos) + 9*2 (sin/cos)

    for obj_idx in range(3):
        i, j = 6*obj_idx, 9*obj_idx
        n_vr_vec[:, j:j+3] = vr_vec[:, i:i+3]
        n_vr_vec[:, j+3:j+9:2] = np.sin(np.radians(vr_vec[:, i+3:i+6]))
        n_vr_vec[:, j+4:j+9:2] = np.cos(np.radians(vr_vec[:, i+3:i+6]))

    vr_data.num_frames = n_vr_vec.shape[0]
    vr_data.frames = n_vr_vec


def convert_mocap_vec_to_sin_cos(mocap_data):
    """
    Converts all degrees in mocap_data to sin/cos values
    :param mocap_data: MocapData obj
    :return: None
    """

    mocap_vec = mocap_data.frames
    n_output_size = (mocap_vec.shape[1]-3)*2 + 3
    n_mocap_vec = np.zeros((mocap_vec.shape[0], n_output_size))

    n_mocap_vec[:, :3] = mocap_vec[:, :3]
    n_mocap_vec[:, 3::2] = np.sin(np.radians(mocap_vec[:, 3:]))
    n_mocap_vec[:, 4::2] = np.cos(np.radians(mocap_vec[:, 3:]))

    mocap_data.num_frames = n_mocap_vec.shape[0]
    mocap_data.frames = n_mocap_vec


def upsample_vr_vec(vr_data, fps):
    """
    Upsamples the data within vr_data to the desired fps
    :param vr_data: VRData object (processed)
    :param fps: desired fps (assumed to be a multiple of vr_fps)
    :return: None
    """

    vr_vec, vr_fps = vr_data.frames, vr_data.fps
    res = fps // vr_fps
    length = (vr_vec.shape[0]-1)*res + 1
    n_vr_vec = np.zeros((length, vr_vec.shape[1]))

    for obj_idx in range(3):
        for subsample in range(res):
            alpha = subsample / res
            i = 6*obj_idx
            pos_x, pos_y = vr_vec[:-1, i:i+3], vr_vec[1:, i:i+3]
            rot_x, rot_y = vr_vec[:-1, i+3:i+6], vr_vec[1:, i+3:i+6]
            n_vr_vec[subsample:-1:res, i:i+3] = lerp_position(pos_x, pos_y, alpha)
            n_vr_vec[subsample:-1:res, i+3:i+6] = lerp_angle(rot_x, rot_y, alpha)
    n_vr_vec[-1] = vr_vec[-1]

    vr_data.num_frames = n_vr_vec.shape[0]
    vr_data.frames = n_vr_vec
    vr_data.fps = fps


def upsample_mocap_vec(mocap_data, fps):
    """
    Upsamples the data within mocap_data to the desired fps
    :param mocap_data: MocapData object
    :param fps: desired fps
    :return: None
    """

    mocap_vec, mocap_fps = mocap_data.frames, mocap_data.fps
    res = fps // mocap_fps
    length = (mocap_vec.shape[0]-1)*res + 1
    n_mocap_vec = np.zeros((length, mocap_vec.shape[1]))

    pos_x, pos_y = mocap_vec[:-1, :3], mocap_vec[1:, :3]
    for subsample in range(res):
        alpha = subsample / res
        n_mocap_vec[subsample:-1:res, :3] = lerp_position(pos_x, pos_y, alpha)
    rot_x, rot_y = mocap_vec[:-1, 3:], mocap_vec[1:, 3:]
    for subsample in range(res):
        alpha = subsample / res
        n_mocap_vec[subsample:-1:res, 3:] = lerp_angle(rot_x, rot_y, alpha)
    n_mocap_vec[-1] = mocap_vec[-1]

    mocap_data.num_frames = n_mocap_vec.shape[0]
    mocap_data.frames = n_mocap_vec
    mocap_data.fps = fps


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


def get_data(time_len):
    """
    Loads all the preprocessed data and turns it into inputs/labels.
    :param time_len: int, sequence length
    :return: inputs, labels, inputs_valid, labels_valid
    """

    # Load and reshape data
    xs, ys = [], []
    for file_name in os.listdir("../data/preprocessed"):
        if file_name == "data-4.txt":
            print("Skipping data-4")
            continue
        mocap_data, vr_data = load_data(file_name)
        xs.append(shape_data(vr_data.frames, time_len))
        ys.append(shape_data(mocap_data.frames, time_len))
    x = np.concatenate(xs, axis=1)
    # y = np.concatenate(ys, axis=1)[:, :, :81]
    y = np.concatenate(ys, axis=1)

    # Shuffle
    indices = np.arange(x.shape[1])
    np.random.shuffle(indices)
    x = x[:, indices, :]
    y = y[:, indices, :]

    # Split into train and valid
    valid_size = x.shape[1] // 10
    return x[:, valid_size:], y[:, valid_size:], x[:, :valid_size], y[:, :valid_size]


def convert_to_predicts(logits):
    """
    Turns logits into angles (as formatted in a standard bvh file)
    :param logits: arr [time_len, 1, output_size]
    :return: arr [time_len, reduced_output_size]
    """

    logits = logits[:, 0, :]
    width = (logits.shape[1]-3)//2 + 3
    predicts = np.zeros((logits.shape[0], width))

    # Switch from meters to centimeters
    predicts[:, :3] = logits[:, :3] * 100
    # Switch from sin/cos to angles (degrees)
    predicts[:, 3:] = np.degrees(np.arctan2(logits[:, 3::2], logits[:, 4::2]))
    return predicts


def save_predicts(predicts, file_name):
    """
    Saves predictions into a file.
    TODO: make this more automatic
    :param predicts: arr [time_len, output_size]
    :param file_name: str
    :return: None
    """

    f = open("exports/" + file_name, "w")
    for i in range(predicts.shape[0]):
        s = " ".join([str(num) for num in predicts[i]])
        f.write(s + "\n")
    f.close()


def main():
    for i in tqdm(range(0, len(data))):
        vr_name, mocap_name, offset = data[i]
        vr_data = load_vr_file("../data/vr/" + vr_name)
        mocap_data = load_mocap_file("../data/mocap/" + mocap_name)
        preprocess_data(mocap_data, vr_data, offset, vr_data.fps)
        save_data(mocap_data, vr_data, "data-" + str(i))


if __name__ == "__main__":
    main()