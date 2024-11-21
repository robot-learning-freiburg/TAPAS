import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def activation_map(softmax_activations, keypoints=None):
    if keypoints is not None:
        keypoints_x, keypoints_y = np.array_split(keypoints.squeeze(0).numpy(), 2)
    for i, density in enumerate(softmax_activations.squeeze(0).numpy()):
        H, W = density.shape
        kp = (keypoints_x[i], keypoints_y[i])

        g = sns.jointplot(data=density, kind="hist")
        g.ax_joint.cla()
        g.ax_marg_y.cla()
        g.ax_marg_x.cla()
        plt.sca(g.ax_joint)
        sns.heatmap(data=density, cbar=False, cmap="Blues")
        # map keypoint to [0, img_size] and scatter it on top
        # need to shift an additional 0.5 to hit center of the pixel
        plt.scatter(
            [(kp[0] + 1) * (W - 1) / 2 + 0.5],
            [(kp[1] + 1) * (H - 1) / 2 + 0.5],
            c="r",
            s=50,
        )

        g.ax_marg_y.barh(np.arange(0.5, H), density.sum(axis=1), color="navy")
        g.ax_marg_x.bar(np.arange(0.5, W), density.sum(axis=0), color="navy")
        # g.ax_marg_y.axhline(density.mean(axis=1).sum(), color='red',
        #                     linewidth=2)
        # g.ax_marg_x.axvline(density.mean(axis=0).sum(), color='red',
        #                     linewidth=2)

        for a in [g.ax_joint, g.ax_marg_x, g.ax_marg_y]:
            a.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                labelbottom=False,
                right=False,
                left=False,
                labelleft=False,
            )

        g.fig.set_size_inches(12, 12)
        g.fig.subplots_adjust(hspace=0.05, wspace=0.02)
        # plt.show()
        g.savefig("distr-{}.png".format(i))
