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
    Reads a stripped down, world coordinate version of a mocap file and returns the raw data.
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
