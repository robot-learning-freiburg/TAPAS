import matplotlib.pyplot as plt
import numpy as np

# from tapas_gmm.utils.multi_processing import mp_wrapper


def make_2d_histos(action_store):
    fig1, ax1 = plt.subplots()
    ax1.hist([action_store[i] for i in range(3)])
    plt.title("Translation distribution")
    plt.show()
    fig2, ax2 = plt.subplots()
    ax2.hist([action_store[i] for i in range(3, 6)])
    plt.title("Rotation distribution")
    plt.show()


def make_3d_histos(action_store):
    n_bins = 25
    cmap = plt.cm.get_cmap("Reds")

    H, (ex, ey, ez) = np.histogramdd([action_store[i] for i in range(3)], bins=n_bins)
    xcenters = (ex[:-1] + ex[1:]) / 2
    ycenters = (ey[:-1] + ey[1:]) / 2
    zcenters = (ez[:-1] + ez[1:]) / 2
    c = H.flatten()
    x = np.repeat(xcenters, n_bins * n_bins)
    y = np.tile(np.repeat(ycenters, n_bins), n_bins)
    z = np.tile(zcenters, n_bins * n_bins)
    idx = c.nonzero()[0]
    c = c[idx]
    x = x[idx]
    y = y[idx]
    z = z[idx]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    img = ax.scatter(x, y, z, c=c, cmap=cmap)
    fig.colorbar(img)
    plt.title("Translation distribution")
    plt.show()

    H, (ex, ey, ez) = np.histogramdd(
        [action_store[i] for i in range(3, 6)], bins=n_bins
    )
    xcenters = (ex[:-1] + ex[1:]) / 2
    ycenters = (ey[:-1] + ey[1:]) / 2
    zcenters = (ez[:-1] + ez[1:]) / 2
    c = H.flatten()
    x = np.repeat(xcenters, n_bins * n_bins)
    y = np.tile(np.repeat(ycenters, n_bins), n_bins)
    z = np.tile(zcenters, n_bins * n_bins)
    idx = c.nonzero()[0]
    c = c[idx]
    x = x[idx]
    y = y[idx]
    z = z[idx]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    img = ax.scatter(x, y, z, c=c, cmap=cmap)
    fig.colorbar(img)
    plt.title("Rotation distribution")
    plt.show()


# @mp_wrapper
def make_all(action_store):
    make_2d_histos(action_store)
    make_3d_histos(action_store)
