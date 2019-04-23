import preprocessing
import numpy as np
import matplotlib.pyplot as plt


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


def get_temporal_offset(mocap_vec, mocap_fps, vr_vec, vr_fps, base_offset, search_window):
    """
    Gets temporal offset that best matches the mocap_vec to the vr_vec.
    It's assumed that mocap_vec is sampled at a higher frequency than vr_vec.
    :param mocap_vec: 1D array of float values
    :param mocap_fps: sample frequency of mocap_vec. Assumed to be a multiple of vr_fps
    :param vr_vec: 1D array of float values
    :param vr_fps: sample frequency of vr_vec
    :param base_offset: starting offset to search around
    :param search_window: how many seconds of offsets to search in both directions
    :return: # of seconds to shift mocap_vec forward
    """

    # Normalize signals
    mocap_vec = (mocap_vec - np.mean(mocap_vec)) / np.std(mocap_vec)
    vr_vec = (vr_vec - np.mean(vr_vec)) / np.std(vr_vec)

    # Compute intermediary values
    skip = mocap_fps // vr_fps
    start_offset = base_offset - mocap_fps*search_window
    end_offset = base_offset + mocap_fps*search_window

    # Iterate over offset and find maximal correlation
    correlations = np.zeros(end_offset - start_offset + 1)
    max_offset, max_correlation = 0, 0
    for offset in range(start_offset, end_offset + 1):
        correlation = get_correlation(vr_vec, mocap_vec, offset, y_skip=skip)
        correlations[offset - start_offset] = correlation
        if correlation > max_correlation:
            max_offset, max_correlation = offset, correlation

    # Plot the correlations
    plt.plot(correlations)
    plt.show()

    # Negate to turn a mocap offset into a vr offset. Division to turn samples to seconds.
    return -max_offset / mocap_fps


def main():
    # Load vr data
    vr_file = "data/vr/2019-18-4--17-36-33.txt"
    vr_data = preprocessing.load_vr_file(vr_file)
    vr_vec = vr_data.frames[:, 1]
    vr_fps = int(np.round(1 / vr_data.time_per_frame))

    # Load mocap data
    mocap_file = "data/mocap/Take 2019-04-18 05.29.41 PM-Head.txt"
    mocap_data = preprocessing.load_mocap_file_helper(open(mocap_file, "r"))
    mocap_vec = mocap_data.frames[:, 1]
    mocap_fps = int(np.round(1 / mocap_data.time_per_frame))

    fps = np.lcm(vr_fps, mocap_fps)
    mocap_vec_upsampled = np.interp(np.arange(0, mocap_vec.size, mocap_fps/fps),
                                    np.arange(mocap_vec.size),
                                    mocap_vec)
    offset = get_temporal_offset(mocap_vec_upsampled, fps, vr_vec, vr_fps, 30, 60)

    print("Offset:", offset, "seconds")

    # Confirm alignment
    plt.plot(np.arange(mocap_vec.size)/mocap_fps + offset, mocap_vec - np.mean(mocap_vec), 'b')
    plt.plot(np.arange(vr_vec.size)/vr_fps, vr_vec - np.mean(vr_vec), 'r')
    plt.show()


if __name__ == "__main__":
    main()
