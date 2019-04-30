"""
Contains utilities for loading and preprocessing data.
"""

import numpy as np


class VRData:

    def __init__(self, num_frames, time_per_frame, calibration_frame, frames):
        self.num_frames = num_frames
        self.time_per_frame = time_per_frame
        self.calibration_frame = calibration_frame
        self.frames = frames


class MocapData:

    def __init__(self, num_frames, time_per_frame, frames):
        # TODO: decide if the skeleton data is required as well
        self.num_frames = num_frames
        self.time_per_frame = time_per_frame
        self.frames = frames


def load_vr_file(file):
    """
    Reads a VR data file and returns the raw data.
    :param file: string
    :return: VRData
    """

    f = open(file, "r")
    num_frames = int(f.readline().split()[1])
    time_per_frame = float(f.readline().split()[2])
    calibration_frame = np.array([float(token) for token in f.readline().split()[1:]])
    num_channels = calibration_frame.size
    frames = np.zeros((num_frames, num_channels))

    for i in range(num_frames):
        frames[i] = np.array([float(token) for token in f.readline().split()])

    return VRData(num_frames, time_per_frame, calibration_frame, frames)


def load_mocap_file_helper(f):
    """
    Reads a a mocap file that's been stripped of its skeleton definition and returns the raw data.
    :param f: file object
    :return: MocapData
    """

    num_frames = int(f.readline().split()[1])
    time_per_frame = float(f.readline().split()[2])
    bookmark = f.tell()
    num_channels = len(f.readline().split())
    f.seek(bookmark)
    frames = np.zeros((num_frames, num_channels))

    for i in range(num_frames):
        frames[i] = np.array([float(token) for token in f.readline().split()])

    return MocapData(num_frames, time_per_frame, frames)


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
