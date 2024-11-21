from math import ceil, sqrt

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import torch
import torchvision.transforms as T
from matplotlib.pyplot import cm

from tapas_gmm.viz.operations import np_channel_front2back, rgb2gray


class InitDbgViz:
    def __init__(self) -> None:
        self.img_dim = None
        self.corr_dim = None

        self.upsampling_factor = None

        self.plot_grid = False

    def plot(
        self,
        depth: tuple[torch.Tensor, ...],
        rgb: tuple[torch.Tensor, ...],
        wc: torch.Tensor,
        mean: torch.Tensor,
        particle_heatmaps: tuple[torch.Tensor, ...],
        keypoints_2d: tuple[torch.Tensor, ...],
        cam_idx: int,
    ) -> None:
        self._setup(depth, rgb, particle_heatmaps)

        self._joint_3d(wc, mean)

        for i in range(self.n_cams):
            img_i = np_channel_front2back(rgb[i].squeeze(0).cpu().numpy())
            self._heatmap_cam(particle_heatmaps[i], keypoints_2d[i], img_i, i)

            self._samples_cam_2d(i, img_i)

            self._sample_cam_3d(i)

        # TODO: don't like that cam_idx construction. Should remove! TODO
        img_i = np_channel_front2back(rgb[cam_idx].squeeze(0).cpu().numpy())

        self._correspondence_kp(
            img_i, keypoints_2d[cam_idx], self.correspondence_diffs[cam_idx]
        )

        self._particles_kp(particle_heatmaps, cam_idx, img_i)

        plt.show()

    def _particles_kp(self, particle_heatmaps, cam_idx, img_i):
        particles_blur = T.functional.gaussian_blur(particle_heatmaps[cam_idx], 9, 2)

        vmin, vmax = particles_blur.min(), particles_blur.max()

        for i in range(self.n_kp):
            ax = self.axes_particles.flat[i]
            ax.imshow(img_i)
            im = ax.imshow(
                particles_blur[0, i], alpha=0.5, cmap="viridis", vmin=vmin, vmax=vmax
            )

        self.fig_particles.subplots_adjust(right=0.8)
        cbar_ax = self.fig_particles.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = self.fig_particles.colorbar(im, cax=cbar_ax)
        cbar.set_label("Particle Density")

    def _correspondence_kp(self, img, keypoints_2d, diffs):
        heatmap = torch.nn.Softmax(dim=2)(diffs.view(1, self.n_kp, -1)).view(
            1, self.n_kp, *self.corr_dim
        )
        heatmap_upsample = torch.nn.functional.upsample(
            heatmap, size=self.img_dim, mode="bilinear", align_corners=True
        ).cpu()[0]

        vmin, vmax = heatmap_upsample.min(), heatmap_upsample.max()

        for i in range(self.n_kp):
            ax = self.axes_perkp.flat[i]
            ax.imshow(img)
            im = ax.imshow(
                heatmap_upsample[i], alpha=0.5, cmap="viridis", vmin=vmin, vmax=vmax
            )
            ax.scatter(
                keypoints_2d[0, i, 0],
                keypoints_2d[0, i, 1],
                edgecolor="r",
                facecolor=None,
                s=self.upsampling_factor,
                marker="s",
            )

        self.fig_perkp.subplots_adjust(right=0.8)
        cbar_ax = self.fig_perkp.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = self.fig_perkp.colorbar(im, cax=cbar_ax)
        cbar.set_label("Correspondence  Probability")

    def _sample_cam_3d(self, i):
        ax = self.axes_joint[1][i]
        ax.set_title(f"Initial 3D samples cam {i}")
        for k in range(self.n_kp):
            ax.scatter(
                self.wc[i, k, :, 0],
                self.wc[i, k, :, 1],
                self.wc[i, k, :, 2],
                c=self.kp_colors[k][None, :],
                s=1,
            )
            ax.scatter(
                self.wc[i, k, :, 0].mean(),
                self.wc[i, k, :, 1].mean(),
                self.wc[i, k, :, 2].mean(),
                c="r",
                marker="X",
            )

    def _samples_cam_2d(self, i, img_i):
        ax = self.axes_joint[0][i]
        ax.set_title(f"Initial 2D samples cam {i}")
        ax.imshow(img_i)
        for k in range(self.n_kp):
            ax.scatter(
                self.samples[i].cpu()[i, k, :, 0],
                self.samples[i].cpu()[i, k, :, 1],
                color=self.kp_colors[k][None, :],
                s=1,
            )

    def _heatmap_cam(self, particle_heatmaps, keypoints_2d, img, i):
        ax = self.axes_joint[2][i]
        ax.set_title(f"Heatmap cam {i}")
        ax.imshow(img)
        ax.imshow(particle_heatmaps[0].mean(dim=0), alpha=0.5)

        if self.plot_grid:
            grid_interval = int(self.img_dim[0] / self.corr_dim[0])
            loc = plticker.MultipleLocator(base=grid_interval)
            ax.xaxis.set_major_locator(loc)
            ax.yaxis.set_major_locator(loc)
            ax.grid(True)

        ax.scatter(keypoints_2d[0, :, 0], keypoints_2d[0, :, 1], color="r")

    def _joint_3d(self, wc, mean):
        ax = self.axes_joint[1][self.n_cams]
        ax.set_title("All cams joint")
        for k in range(self.n_kp):
            ax.scatter(
                wc[0, k, :, 0],
                wc[0, k, :, 1],
                wc[0, k, :, 2],
                c=self.kp_colors[k][None, :],
                s=1,
            )
            ax.scatter(mean[0, k, 0], mean[0, k, 1], mean[0, k, 2], c="r", marker="X")

    def _setup(self, depth, rgb, particle_heatmaps):
        if self.img_dim is None:
            self.img_dim = rgb[0].shape[-2:]

        if self.corr_dim is None:
            self.corr_dim = self.correspondence_diffs[0].shape[-2:]

        if self.upsampling_factor is None:
            self.upsampling_factor = self.img_dim[0] / self.corr_dim[0]

        n_cams = len(depth)
        n_kp = particle_heatmaps[0].shape[1]

        self.kp_colors = cm.rainbow(np.linspace(0, 1, n_kp))

        n_rows = n_cols = ceil(sqrt(n_kp))

        self.fig_joint, self.axes_joint = plt.subplots(
            3, n_cams + 1, figsize=(19.2, 14.4)
        )
        self.fig_perkp, self.axes_perkp = plt.subplots(
            n_rows, n_cols, figsize=(19.2, 14.4)
        )
        self.fig_particles, self.axes_particles = plt.subplots(
            n_rows, n_cols, figsize=(19.2, 14.4)
        )

        if n_kp == 1:
            self.axes_perkp = np.expand_dims(self.axes_perkp, axis=0)

        self.fig_joint.suptitle("Joint Heatmaps and Init Prediction")
        self.fig_perkp.suptitle("Per Keypoint Heatmaps and Init Prediction")
        self.fig_particles.suptitle("Per KeyPoint Init Particles")

        for i in range(n_cams + 1):
            ax = self.axes_joint[1][i]
            sps = ax.get_subplotspec()
            ax.remove()
            self.axes_joint[1][i] = self.fig_joint.add_subplot(sps, projection="3d")

        self.axes_joint[0][n_cams].remove()
        self.axes_joint[2][n_cams].remove()

        self.n_cams = n_cams
        self.n_kp = n_kp

    def add_samples(
        self, diffs: tuple[torch.Tensor, ...], samples: tuple[torch.Tensor, ...]
    ) -> None:
        self.correspondence_diffs = diffs
        self.samples = samples

    def add_world_coordinates(self, wc: torch.Tensor) -> None:
        self.wc = wc


class SampleObsViz:
    def __init__(self) -> None:
        self.fig = None
        self.axes = None

        self.n_kp = None

    def update(
        self,
        rgb: tuple[torch.Tensor],
        diffs: tuple[torch.Tensor, ...],
        samples: tuple[torch.Tensor, ...],
    ) -> None:
        diffs = diffs[0]  # TODO: multicam!

        if self.fig is None:
            self.n_kp = diffs.shape[1]

            n_rows = n_cols = ceil(sqrt(self.n_kp))
            self.fig, self.axes = plt.subplots(n_rows, n_cols, figsize=(19.2, 14.4))

        H, W = rgb[0].shape[-2:]
        corr_dim = diffs.shape[-2:]
        n_kp = diffs.shape[-3]

        heatmap = torch.nn.Softmax(dim=2)(diffs.view(1, n_kp, -1)).view(
            1, n_kp, *corr_dim
        )

        heatmap_upsample = torch.nn.functional.upsample(
            heatmap.cpu(), size=[H, W], mode="bilinear", align_corners=True
        )

        for i in range(self.n_kp):
            self.axes.flat[i].clear()
            img_i = np_channel_front2back(rgb[0].squeeze(0).cpu().numpy())
            self.axes.flat[i].imshow(img_i)
            self.axes.flat[i].imshow(
                heatmap_upsample[0, i].numpy(), interpolation="none", alpha=0.5
            )
            self.axes.flat[i].scatter(
                samples[0].cpu()[0, i, :, 0], samples[0].cpu()[0, i, :, 1], color="r"
            )

        plt.show()


class ParticleFilterViz:
    def __init__(self):
        self.batch = [0, 0, 0, 0]
        self.keypoints = [0, 1, 8, 9]
        self.n_rows = len(self.batch)
        self.n_cols = 3

        self.fig_names = ["kp{}".format(str(i)) for i in range(len(self.keypoints))]

        self.cbars = []

        self.traj_counter = 0
        self.obs_counter = 0
        self.file = "src/_tmp/pf-"

        self.show = False

    def reset_episode(self):
        self.traj_counter += 1
        self.obs_counter = 0

    def run(self):
        self.fig, self.axes = plt.subplots(
            self.n_rows, self.n_cols, figsize=(19.2, 14.4)
        )

        for i in range(self.n_rows):
            sps = self.axes[i][0].get_subplotspec()
            self.axes[i][0].remove()
            self.axes[i][0] = self.fig.add_subplot(sps, projection="3d")

        for i in range(self.n_rows):  # fill some dummy value to setup color bar
            ax = self.axes[i][0]
            scp = ax.scatter(0.5, 0.5, 0.5, c=0.5, cmap="hot_r")
            # scp.set_clim(0, 1)
            cb = plt.colorbar(scp, ax=ax)

            self.cbars.append(cb)

        plt.ion()
        plt.tight_layout()

        if self.show:
            plt.show()

    def update(self, coordinates, weights, prediction, heatmaps, keypoints_2d, rgbs):
        prediction = torch.stack(torch.chunk(prediction, 3, dim=1), dim=2)

        for i in range(self.n_rows):
            ax = self.axes[i][0]
            self.cbars[i].remove()
            ax.clear()

            b = self.batch[i]
            k = self.keypoints[i]

            c = coordinates[b][k]
            w = weights[b][k]
            kp3d = prediction[b][k]

            scp = ax.scatter(c[..., 0], c[..., 1], c[..., 2], c=w, cmap="hot_r")

            self.cbars[i] = plt.colorbar(scp, ax=ax)

            ax.scatter(kp3d[..., 0], kp3d[..., 1], kp3d[..., 2], marker="X", c="b")

            for j, (h, r, kp2d) in enumerate(
                zip(heatmaps, rgbs, keypoints_2d)
            ):  # noqa 501
                ax = self.axes[i][j + 1]
                ax.clear()
                img = rgb2gray(np_channel_front2back(r[b].numpy()))
                ax.imshow(img, cmap="gray", alpha=0.3, interpolation="none")
                ax.imshow(h[b][k], alpha=0.7, cmap="hot_r", interpolation="none")
                ax.scatter(kp2d[b][k][0], kp2d[b][k][1], marker="X", c="b")
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        file = self.file + str(self.traj_counter) + "-" + str(self.obs_counter) + ".png"
        plt.savefig(file)

        self.obs_counter += 1

        if self.show:
            plt.pause(0.001)
            self.fig.canvas.draw()
