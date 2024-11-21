import itertools
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T

# import matplotlib
from loguru import logger
from matplotlib import cm

from tapas_gmm.encoder.keypoints import PriorTypes
from tapas_gmm.viz.operations import channel_front2back, rgb2gray, scale


class VizType(Enum):
    IMG = 1
    DESCRIPTOR = 2
    DEPTH = 3
    IMG_W_SPP = 4  # KP: img, sm, prior, post
    IMG_W_TKP = 5  # Transporter KPs on image
    SPP_PER_KP = 6
    PARTICLE = 7


class LiveKeypoints:
    def __init__(
        self,
        threshold=0.125,
        fig_names=("1",),
        n_rows=1,
        cams=None,
        viz=None,
        n_figs=None,
        channels_to_show=None,
        contour=True,
    ):
        self.threshold = threshold or np.inf
        self.fig_names = fig_names
        self.n_rows = n_rows

        self.n_figs = n_figs
        self.viz = viz
        self.channels_to_show = channels_to_show
        self.cams = cams
        self.n_cams = len(cams) if cams is not None else 0

        self.contour = False
        self.sum = True

        self._height_history = []

    def reset(self):
        self._height_history = []

    def run(self):
        n_cols = int(len(self.fig_names) / self.n_rows)
        self.fig, self.axes = plt.subplots(self.n_rows, n_cols)
        if hasattr(self.axes, "flat"):
            axes = self.axes.flat
        else:
            axes = [self.axes]
        for i, ax in enumerate(axes):
            ax.axis("off")
            ax.set_title(self.fig_names[i])
        plt.ion()
        plt.tight_layout()
        plt.show()

    def update(
        self, ax_no, keypoints, embedding, dist, best, particles, world_coords=None
    ):
        if hasattr(self.axes, "flat"):
            ax = self.axes.flat[ax_no]
        else:
            ax = self.axes
        ax.clear()
        ax.set_title(self.fig_names[ax_no])
        if world_coords is None:
            ax.axis("off")
        else:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        if particles is not None:
            embedding = channel_front2back(embedding.squeeze(0))
            img = scale(rgb2gray(embedding.numpy()).astype(np.float32))
            ax.imshow(img, cmap="gray", alpha=0.3, interpolation="none")
            # TODO: how to plot particles? One color per kp?
            particles = torch.sum(T.functional.gaussian_blur(particles[0], 9, 4), dim=0)
            ax.imshow(particles, interpolation="none", alpha=0.5)
            # ax.imshow(particles.squeeze(0), alpha=0.7, cmap='hot_r',
            #           interpolation='none')
            keypoints = keypoints.squeeze(0).numpy()
            keypoints_x, keypoints_y = keypoints[:, 0], keypoints[:, 1]
            ax.scatter(keypoints_x, keypoints_y, c="r")
            for i, txt in enumerate(range(keypoints.shape[0])):
                ax.annotate(str(i), (keypoints_x[i], keypoints_y[i]))

        elif world_coords is not None:
            self._height_history.append(world_coords[0, :, 2].cpu().numpy())
            for j in range(16):
                ax.plot(
                    list(range(len(self._height_history))),
                    [h[j] for h in self._height_history],
                )
        else:
            if len(embedding.shape) == 3:  # images
                embedding = channel_front2back(embedding)
            if len(embedding.shape) == 4:  # per kp-heatmaps
                if self.sum:
                    img = embedding.squeeze(0).sum(dim=0)
                    H, W = img.shape[:2]
                    ax.imshow(img)

                if self.contour:
                    N, H, W = embedding.shape[1:]
                    X = np.arange(0, H, 1)
                    Y = np.arange(0, W, 1)
                    X, Y = np.meshgrid(X, Y)
                    cmap = cm.get_cmap("tab20", N).colors
                    for e, c in zip(embedding[0], cmap):
                        ax.contour(X, Y, e, 0, colors=[c[:3]], linewidths=1)
                        ax.set_aspect("equal")
                        # TODO: custom mono-colored (saturation changing)
                        # colormap for each img show.
                        # cdict = {}
                        # smap = matplotlib.colors.LinearSegmentedColormap(
                        #     'custom_cmap', cdict)
                        # ax.imshow(e, cmap=smap)

            else:  # images or single-channel heatmaps
                img = scale(embedding.numpy()).astype(np.float32)
                H, W = img.shape[:2]
                ax.imshow(img)

            # TODO: make work for 3D keypoints as well
            # Points are normalized to [-1, 1]., so map to [0, img_size - 1].
            # spatial_expectation_of_reference_descriptors returns (x, y) order.
            keypoints_x, keypoints_y = np.array_split(
                keypoints.squeeze(0).cpu().numpy(), 2
            )
            pos_x = [np.round((i + 1) * (W - 1) / 2) for i in keypoints_x]
            pos_y = [np.round((i + 1) * (H - 1) / 2) for i in keypoints_y]

            n_points = keypoints_x.shape[0]

            if dist is None or best is None:
                colors = "r"
            else:
                colors = [
                    "w" if not b else "r" if d < self.threshold else "b"
                    for d, b in zip(dist, best)
                ]
            # print(dist)
            ax.scatter(pos_x, pos_y, c=colors)
            for i, txt in enumerate(range(n_points)):
                ax.annotate(str(i), (pos_x[i], pos_y[i]))

        plt.pause(0.001)
        self.fig.canvas.draw()

    @classmethod
    def setup_from_conf(cls, config, policy):
        channels_to_show = None

        if config.eval.viz:
            try:  # HACK
                policy.encoder.particle_filter.running_eval = True
                logger.info("Enabled viz for particle filter.")
            except Exception:
                pass

        if (
            hasattr(config.policy, "encoder_name")
            and config.policy.encoder_name
            in ["keypoints", "keypoints_gt", "transporter"]
            and config.eval.viz
        ):
            n_channel = 1
            if config.policy.encoder_name == "keypoints_gt":
                viz = VizType.IMG
                fig_types = ["cam"]
            elif config.policy.encoder_name == "transporter":
                viz = VizType.IMG_W_TKP
                fig_types = ["cam"]
            elif (
                config.policy.encoder_config.encoder.prior_type
                is PriorTypes.PARTICLE_FILTER
            ):
                viz = VizType.PARTICLE
                fig_types = ["cam"]
            elif config.policy.encoder_config.encoder.descriptor_dim == 3:
                viz = VizType.DESCRIPTOR
                fig_types = ["cam", "descriptor"]
            elif config.eval.kp_per_channel_viz:
                viz = VizType.SPP_PER_KP
                n_kp = config.policy.encoder_config.encoder.keypoints.n_sample
                if config.eval.show_channels is None:
                    channels_to_show = list(range(n_kp))
                else:
                    channels_to_show = config.eval.show_channels
                n_channel = len(channels_to_show)
                fig_types = [
                    "{} kp {}".format(f, k)
                    for k in channels_to_show
                    for f in ("prior", "sm", "post")
                ]
            else:
                viz = VizType.IMG_W_SPP  # IMG
                fig_types = ["cam", "prior", "sm", "post"]

            threshold = config.policy.encoder_config.threshold_keypoint_dist

            cams = config.env_config.cameras
            n_cams = len(cams)

            fig_names = [
                "{} cam {}".format(j, i) for i in range(n_cams) for j in fig_types
            ] + (["kp height"] if viz == VizType.PARTICLE else [])

            n_rows = n_cams * n_channel + (1 if viz == VizType.PARTICLE else 0)

            keypoint_viz = cls(
                threshold,
                fig_names,
                n_rows,
                cams,
                viz,
                len(fig_types),
                channels_to_show,
            )

            keypoint_viz.run()

        else:
            keypoint_viz = None

        return keypoint_viz

    def update_from_info(self, info, obs):
        particles = [None for _ in range(self.n_cams) for _ in range(self.n_figs)]

        cam_obs_map = {"wrist": obs.cam_w_rgb, "overhead": obs.cam_o_rgb}

        if self.viz in [
            VizType.DESCRIPTOR,
            VizType.DEPTH,
            VizType.IMG,
            VizType.IMG_W_SPP,
            VizType.SPP_PER_KP,
        ]:
            encodings = info["kp_raw_2d"]
            distance = info["distance"]

            encodings = [t.squeeze(0) for t in encodings for _ in range(self.n_figs)]
            distance = [
                t.squeeze(0) if t is not None else None
                for t in distance
                for _ in range(self.n_figs)
            ]

            if distance[0] is not None:
                dist_per_cam = torch.stack(distance)
                best_cam = torch.argmin(dist_per_cam, dim=0)
            else:
                dist_per_cam = [None for _ in range(self.n_cams)]
                best_cam = None
        if self.viz is VizType.DESCRIPTOR:
            embeddings = info["descriptor"]
        elif self.viz is VizType.DEPTH:
            embeddings = info["depth"]
        elif self.viz is VizType.IMG:
            embeddings = tuple((torch.from_numpy(cam_obs_map[c]) for c in self.cams))
        elif self.viz is VizType.IMG_W_SPP:
            if info["prior"][0] is None:
                prior = tuple((torch.empty(1, 16, 32, 32) for _ in range(self.n_cams)))
            else:
                prior = info["prior"]
            embeddings = list(
                itertools.chain(
                    *zip(
                        [torch.from_numpy(cam_obs_map[c]) for c in self.cams],
                        prior,
                        info["sm"],
                        info["post"],
                    )
                )
            )
            embeddings = [e.cpu() for e in embeddings]
        elif self.viz is VizType.SPP_PER_KP:
            if info["prior"][0] is None:
                prior = tuple((torch.empty(1, 16, 32, 32) for _ in range(self.n_cams)))
            else:
                prior = info["prior"]
            sm, post = info["sm"], info["post"]
            n_kp = sm[0].shape[1]
            n_emb = 3
            embeddings = [
                e[c][0][k].unsqueeze(0).cpu()
                for c in range(self.n_cams)
                for k in range(n_kp)
                for e in (prior, sm, post)
                if k in self.channels_to_show
            ]
            # get the respective keypoint
            channel = [c for c in self.channels_to_show for _ in range(n_emb)]
            encodings = [
                e.cpu().index_select(0, torch.tensor([i, i + n_kp]))
                for i, e in zip(channel, encodings)
            ]
            distance = [
                d[i % n_kp].unsqueeze(0) if d is not None else None
                for i, d in zip(channel, distance)
            ]
        elif self.viz is VizType.IMG_W_TKP:
            embeddings = tuple((torch.from_numpy(cam_obs_map[c]) for c in self.cams))
            embeddings = [info["heatmap"]]
            encodings = info["vis_encoding"].squeeze(0).chunk(self.n_cams)

            distance = [None for c in range(self.n_cams)]
            best_cam = None
        elif self.viz is VizType.PARTICLE:
            particles = info["particles_2d"]
            embeddings = tuple((torch.from_numpy(cam_obs_map[c]) for c in self.cams))
            # encodings = tuple(torch.cat((e[0, ..., 0], e[0, ..., 1]))
            #                   for e in info["keypoints_2d"])
            encodings = info["keypoints_2d"]
            distance = [None for c in range(self.n_cams)]
            best_cam = None
            world_coords = info["world_coordinates"]
        else:
            pass

        cam_nos = [c for c in range(self.n_cams) for _ in range(self.n_figs)]
        for k, (enc, emb, dist, i) in enumerate(
            zip(encodings, embeddings, distance, cam_nos)
        ):
            if best_cam is None:
                best = None
            elif self.viz is VizType.SPP_PER_KP:
                c = self.channels_to_show[k // n_emb]
                best = best_cam[c % n_kp].unsqueeze(0) == i
            else:
                best = best_cam == i
            self.update(k, enc, emb, dist, best, particles=particles[k])
        if self.viz is VizType.PARTICLE:
            self.update(-1, None, None, None, None, None, world_coords=world_coords)
