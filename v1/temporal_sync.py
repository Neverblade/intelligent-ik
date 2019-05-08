from v1 import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import math
import getopt
import sys

vr_file = ""
mocap_file = ""


def get_correlation(x, y, offset, x_skip=1, y_skip=1, plot=False):
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
    x = x[::x_skip]
    y = y[::y_skip]
    min_length = min(x.size, y.size)
    x = x[:min_length]
    y = y[:min_length]
    return np.dot(x, y)


def get_temporal_offset(mocap_vec, mocap_fps, vr_vec, vr_fps, base_offset, search_window, start_time=None, end_time=None):
    """
    Gets temporal offset that best matches the mocap_vec to the vr_vec.
    It's assumed that mocap_vec is sampled at a higher frequency than vr_vec.
    :param mocap_vec: 1D array of float values
    :param mocap_fps: sample frequency of mocap_vec. Assumed to be a multiple of vr_fps
    :param vr_vec: 1D array of float values
    :param vr_fps: sample frequency of vr_vec
    :param base_offset: starting offset to search around, in seconds
    :param search_window: how many seconds of offsets to search in both directions
    :param start_time: timestamp in mocap signal to start including in correlation
    :param end_time: timestamp in mocap signal to stop including in correlation
    :return: # of seconds to shift mocap_vec forward
    """

    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = 100000

    # Remove samples past end_time
    vr_vec = vr_vec[start_time*vr_fps:end_time*vr_fps]

    # Compute intermediary values. Negate base_offset to turn mocap offset into vr offset
    skip = mocap_fps // vr_fps
    start_offset = int(mocap_fps * (-base_offset - search_window))
    end_offset = int(mocap_fps * (-base_offset + search_window))
    start_time_offset = start_time*mocap_fps

    # Iterate over offset and find maximal correlation
    correlations = np.zeros(end_offset - start_offset + 1)
    max_offset, max_correlation = None, -math.inf
    for offset in range(start_offset, end_offset + 1):
        correlation = get_correlation(vr_vec, mocap_vec, offset + start_time_offset, y_skip=skip)
        correlations[offset - start_offset] = correlation
        if correlation > max_correlation:
            max_offset, max_correlation = offset, correlation

    #plt.plot(correlations)
    #plt.show()

    # Negate to turn vr offset into mocap offset.
    return -max_offset/mocap_fps


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
    vr_fps = int(np.round(1 / vr_data.time_per_frame))
    vr_vec = (vr_vec - np.mean(vr_vec)) / np.std(vr_vec)

    # Load mocap data
    mocap_data = preprocessing.load_mocap_file_helper(open(mocap_file, "r"))
    mocap_vec = mocap_data.frames[:, 1]
    mocap_fps = int(np.round(1 / mocap_data.time_per_frame))
    mocap_vec = (mocap_vec - np.mean(mocap_vec)) / np.std(mocap_vec)

    """
    # Render and grab threshold
    plt.plot(vr_vec, "r")
    plt.plot(mocap_vec, "b")
    plt.show()
    threshold = float(input("Threshold: "))

    # 0-out entries below threshold
    vr_vec[np.abs(vr_vec) < threshold] = 0
    mocap_vec[np.abs(mocap_vec) < threshold] = 0

    plt.plot(vr_vec, "r")
    plt.plot(mocap_vec, "b")
    plt.show()
    """

    # Compute shared fps and the upsampled mocap_vec
    fps = np.lcm(vr_fps, mocap_fps)
    mocap_vec_upsampled = np.interp(np.arange(0, mocap_vec.size, mocap_fps/fps),
                                    np.arange(mocap_vec.size),
                                    mocap_vec)

    predicted_offset = get_temporal_offset(mocap_vec_upsampled, fps, vr_vec, vr_fps, 0, 30, start_time=200, end_time=1200)
    start, end, window, gap = 0, 60, 1, 30
    offsets = []
    while end < vr_vec.size / vr_fps:
        offset = get_temporal_offset(mocap_vec_upsampled, fps, vr_vec, vr_fps, predicted_offset, window, start_time=start, end_time=end)
        offsets.append(offset)
        start += gap
        end += gap

    print("Predicted Offset:", predicted_offset)
    variability = np.array(offsets) - predicted_offset
    print("Variability:", variability)
    print("Total variability:", np.sum(np.abs(variability)))

    # Confirm alignment through plot
    mocap_vec = (mocap_vec - np.mean(mocap_vec)) / np.std(mocap_vec)
    vr_vec = (vr_vec - np.mean(vr_vec)) / np.std(vr_vec)

    plt.plot(np.arange(mocap_vec.size)/mocap_fps + predicted_offset, mocap_vec, 'b')
    plt.plot(np.arange(vr_vec.size)/vr_fps, vr_vec, 'r')
    plt.show()


if __name__ == "__main__":
    main()
