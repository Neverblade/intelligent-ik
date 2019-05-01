"""
Contains utilities for loading and preprocessing data.
"""

import numpy as np


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


def preprocess_data(mocap_data, vr_data, offset, desired_fps):
    """
    Preprocesses mocap_data and vr_data. Performs the offset and sampling required.
    :param mocap_data: MocapData object
    :param vr_data: VRData object (processed)
    :param offset: temporal offset - move mocap_data forward in time.
    :param desired_fps: fps each sequence should end up at. assumed to divisble under lcm of fps's
    :return: None
    """

    # Upsample both
    mocap_fps, vr_fps = mocap_data.fps, vr_data.fps
    fps = np.lcm(mocap_fps, vr_fps)
    upsample_mocap_vec(mocap_data, fps)
    upsample_vr_vec(vr_data, fps)
    mocap_vec, vr_vec = mocap_data.frames, vr_data.frames

    # Apply offset and downsample
    sample_offset, skip = fps*offset, fps//desired_fps
    mocap_vec = mocap_vec[max(-sample_offset, 0):]
    vr_vec = vr_vec[max(sample_offset, 0):]
    min_length = min(mocap_vec.shape[0], vr_vec.shape[0])
    mocap_vec = mocap_vec[:min_length:skip]
    vr_vec = vr_vec[:min_length:skip]
    mocap_data.num_frames, mocap_data.frames = mocap_vec.shape[0], mocap_vec
    vr_data.num_frames, vr_data.frames = vr_vec.shape[0], vr_vec
    vr_data.fps, mocap_data.fps = desired_fps, desired_fps

    # Convert vr inputs to sin/cos
    convert_vr_vec_to_sin_cos(vr_data)

    # Scale down positions of mocap (centimeters to meters)
    mocap_data.frames[:, :3] /= 100


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
        n_vr_vec[:, j+3:j+9:2] = np.sin(vr_vec[:, i+3:i+6])
        n_vr_vec[:, j+4:j+9:2] = np.cos(vr_vec[:, i+3:i+6])

    vr_data.num_frames = n_vr_vec.shape[0]
    vr_data.frames = n_vr_vec


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


# Testing lerp_angle
# print(lerp_angle(0, 30, .5))
# print(lerp_angle(40, 190, 1))
# print(lerp_angle(0, 180, .75))

# Testing lerp position
# print(lerp_position(0, 20, .5))
# print(lerp_position(-23, 7, .25))
# print(lerp_position(0, 1, 1))
#print(lerp_position(np.arange(10).reshape((2,5)), np.arange(10).reshape((2, 5)) + 5, .25))

# Testing upsample vr vec
# vr_data = load_vr_file("data/vr/2019-16-4--16-24-35-processed.txt")
# upsample_vr_vec(vr_data, 100)
# print(vr_data.num_frames)
# print(vr_data.frames.shape)
# print(vr_data.frames[:10, 2:4])

# Testing upsample mocap vec
# mocap_data = load_mocap_file("data/mocap/Take 2019-04-16 04.19.31 PM.bvh")
# upsample_mocap_vec(mocap_data, 360)
# print(mocap_data.num_frames)
# print(mocap_data.frames.shape)
# print(mocap_data.frames[:20,8])

# Testing sin/cos conversion
# vr_data = load_vr_file("data/vr/2019-16-4--16-24-35-processed.txt")
# convert_vr_vec_to_sin_cos(vr_data)
# print(vr_data.num_frames, vr_data.frames.shape)
# print(vr_data.frames[:20, 9:18])

# Testing preprocess
# vr_data = load_vr_file("data/vr/2019-16-4--16-24-35-processed.txt")
# mocap_data = load_mocap_file("data/mocap/Take 2019-04-16 04.19.31 PM.bvh")
# preprocess_data(mocap_data, vr_data, 1, vr_data.fps)
# print(vr_data.num_frames, mocap_data.num_frames)
# print(vr_data.fps, mocap_data.fps)
# print(vr_data.frames.shape, mocap_data.frames.shape)
# print(vr_data.frames[0])
# print(mocap_data.frames[0])
#
# import matplotlib.pyplot as plt
# plt.plot(np.arange(vr_data.num_frames)/vr_data.fps, vr_data.frames[:,1], "r")
# plt.plot(np.arange(vr_data.num_frames)/vr_data.fps, mocap_data.frames[:,1], "b")
# plt.show()