import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import math
import getopt
import sys

vr_file = ""
mocap_file = ""


def get_correlation(x, y, offset):
    """
    Finds the correlation between x and y at the given offset.
    x is shifted forward by the offset.
    :param x: 1D array of float values
    :param y: 1D array of float values
    :param offset: how much to offset x by
    :param x_skip: take every x_skip samples in x
    :param y_skip: take every y_skip samples in y
    :return: float representing degree of correlation
    """

    x = x[max(-offset, 0):]
    y = y[max(offset, 0):]
    min_length = min(x.size, y.size)
    x = x[:min_length]
    y = y[:min_length]
    return np.dot(x, y)


def get_temporal_offset(x, y, fps, base, window, start=None, end=None):
    """
    Finds best temporal offset aligning x and y.
    :param x: np array
    :param y: np array
    :param fps: sample rate of x and y
    :param base: seconds; starting rate to search around
    :param window: seconds; search radius
    :param start: starting time in x to consider correlation
    :param end: ending time in x to consider correlation
    :return: seconds to shift x forward in time
    """

    start_sample = int(fps * (0 if start is None else start))
    end_sample = int(fps * (1000000 if end is None else end))
    x = x[start_sample:end_sample]

    start_sample_offset, end_sample_offset = int(fps*(base-window)), int(fps*(base+window))
    correlations = np.zeros(end_sample_offset - start_sample_offset + 1)
    max_sample_offset, max_correlation = None, -np.inf

    for sample_offset in range(start_sample_offset, end_sample_offset+1):
        correlation = get_correlation(x, y, sample_offset + start_sample)
        correlations[sample_offset-start_sample_offset] = correlation
        if correlation > max_correlation:
            max_sample_offset, max_correlation = sample_offset, correlation

    return max_sample_offset / fps


def process_args():
    options = "v:m:"
    long_options = ["vr=", "mocap="]
    try:
        opts, args = getopt.getopt(sys.argv[1:], options, long_options)
    except getopt.GetoptError:
        print("temporal_sync.py -v <vr> -m <mocap>")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-v", "--vr"):
            global vr_file
            vr_file = arg
        elif opt in ("-m", "--mocap"):
            global mocap_file
            mocap_file = arg


def main():
    # Command line arguments
    process_args()

    # Load vr data
    vr_data = preprocessing.load_vr_file(vr_file)
    vr_vec = vr_data.frames[:, 1]
    vr_fps = vr_data.fps
    vr_vec = (vr_vec - np.mean(vr_vec)) / np.std(vr_vec)

    # Load mocap data
    mocap_data = preprocessing.load_mocap_file_helper(open(mocap_file, "r"))
    mocap_vec = mocap_data.frames[:, 1]
    mocap_fps = mocap_data.fps
    mocap_vec = (mocap_vec - np.mean(mocap_vec)) / np.std(mocap_vec)

    # Upsample both sequences
    fps = np.lcm(vr_fps, mocap_fps)
    vr_vec_upsampled = np.interp(np.arange(0, vr_vec.size, vr_fps/fps),
                                 np.arange(vr_vec.size),
                                 vr_vec)
    mocap_vec_upsampled = np.interp(np.arange(0, mocap_vec.size, mocap_fps/fps),
                                    np.arange(mocap_vec.size),
                                    mocap_vec)

    predicted_offset = get_temporal_offset(mocap_vec_upsampled, vr_vec_upsampled, fps, 0, 10)
    print("Predicted Offset:", predicted_offset)

    #"""
    start, end, window, gap = 0, 60, 1, 30
    offsets = []
    while end < vr_vec.size / vr_fps:
        offset = get_temporal_offset(mocap_vec_upsampled, vr_vec_upsampled, fps,
                                     predicted_offset, 10,
                                     start=start, end=end)
        offsets.append(offset)
        start, end = start + gap, end + gap

    variability = np.array(offsets) - predicted_offset
    print("Variability:", variability)
    print("Average variability: ", np.mean(variability))
    #"""

    plt.plot(np.arange(mocap_vec.size) / mocap_fps + predicted_offset, mocap_vec, 'b')
    plt.plot(np.arange(vr_vec.size) / vr_fps, vr_vec, 'r')
    plt.show()


if __name__ == "__main__":
    main()
