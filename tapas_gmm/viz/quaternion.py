import matplotlib.pyplot as plt
import numpy as np


def plot_data(data, min_idx=3, max_idx=7, time_based=False):
    quat = data[..., min_idx:max_idx]
    if time_based:
        plot_quat_time_based(quat)
    else:
        plot_quat_imaginary_3d(quat)


def plot_quat_time_based(quat):
    quat_shape = quat.shape
    n_trajs = quat_shape[0]
    fig, ax = plt.subplots(1, 4)
    fig.set_size_inches(16, 4)
    for i in range(n_trajs):
        ax[0].plot(quat[i, ..., 0])
        ax[1].plot(quat[i, ..., 1])
        ax[2].plot(quat[i, ..., 2])
        ax[3].plot(quat[i, ..., 3])
    plt.show()


def plot_quat_imaginary_3d(quat):
    quat_shape = quat.shape
    n_trajs = quat_shape[0]
    if len(quat_shape) == 3:
        fig, ax = plt.subplots(n_trajs, subplot_kw=dict(projection="3d"))
        fig.set_size_inches(4, n_trajs * 4)
        for i in range(n_trajs):
            ax[i].plot(quat[i, ..., 1], quat[i, ..., 2], quat[i, ..., 3])
    elif len(quat_shape) == 4:
        n_frames = quat_shape[2]
        fig, ax = plt.subplots(n_trajs, n_frames, subplot_kw=dict(projection="3d"))
        fig.set_size_inches(4 * n_frames, n_trajs * 4)
        for i in range(n_trajs):
            for j in range(n_frames):
                ax[i, j].scatter(quat[i, :, j, 1], quat[i, :, j, 2], quat[i, :, j, 3])
    plt.show()


def plot_quat_components(quat):
    assert len(quat.shape) == 4
    n_trajs, n_times, n_frames, n_components = quat.shape
    assert n_components == 4

    fig, ax = plt.subplots(n_trajs, n_frames)
    if n_frames == 1:
        ax = ax[:, np.newaxis]
    fig.set_size_inches(4 * n_frames, 4 * n_trajs)
    for i in range(n_trajs):
        for j in range(n_frames):
            for k in range(n_components):
                ax[i, j].plot(quat[i, :, j, k])
                frame_type = "pos" if n_frames == 1 else "vel" if j % 2 == 1 else "pos"
                frame_no = 0 if n_frames == 1 else j // 2
                ax[i, j].set_title(f"Frame {frame_no} ({frame_type})")

    plt.show()
