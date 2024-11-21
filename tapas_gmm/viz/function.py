# from tapas_gmm.utils.multi_processing import mp_wrapper
import matplotlib.pyplot as plt
import numpy as np


# @mp_wrapper
def show_function(f, xmin, xmax):
    x = np.arange(xmin, xmax, (xmax - xmin) / 100)
    y = np.array([f(i) for i in x])
    fig1, ax1 = plt.subplots()
    ax1.scatter(x, y)
    plt.show()
