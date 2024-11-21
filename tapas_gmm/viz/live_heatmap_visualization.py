import random
from dataclasses import dataclass

import cv2

# import matplotlib.pyplot as plt
import numpy as np

import tapas_gmm.dense_correspondence.correspondence_finder as correspondence_finder
from tapas_gmm.dataset.dc import DenseCorrespondenceDataset
from tapas_gmm.encoder import KeypointsPredictor, VitFeatureEncoder
from tapas_gmm.viz.operations import channel_front2back

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLACK = (0, 0, 0)


def draw_reticle(img, u, v, label_color):
    """
    Draws a reticle on the image at the given (u,v) position

    :param img:
    :type img:
    :param u:
    :type u:
    :param v:
    :type v:
    :param label_color:
    :type label_color:
    :return:
    :rtype:
    """
    white = (255, 255, 255)
    cv2.circle(img, (u, v), 4, label_color, 1)
    cv2.circle(img, (u, v), 5, white, 1)
    cv2.circle(img, (u, v), 6, label_color, 1)
    cv2.line(img, (u, v + 1), (u, v + 2), white, 1)
    cv2.line(img, (u + 1, v), (u + 2, v), white, 1)
    cv2.line(img, (u, v - 1), (u, v - 2), white, 1)
    cv2.line(img, (u - 1, v), (u - 2, v), white, 1)


def compute_gaussian_kernel_heatmap_from_norm_diffs(norm_diffs, taper, norm):
    """
    Computes and RGB heatmap from norm diffs
    :param norm_diffs: distances in descriptor space to a given keypoint
    :type norm_diffs: numpy array of shape [H,W]
    :param variance: the variance of the kernel
    :type variance:
    :return: RGB image [H,W,3]
    :rtype:
    """
    heatmap = np.copy(norm_diffs)

    heatmap = np.exp(-heatmap * taper)  # these are now in [0,1]

    if norm:
        heatmap = heatmap / np.mean(heatmap)

    heatmap *= 255
    heatmap = heatmap.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap_color


def display(*rows, scale_factor, window_name):
    # n_rows = len(rows)
    # n_cols = max([len(r) for r in rows])
    #
    # matrix = np.zeros((n_rows*img_height, n_cols*img_width, 3))
    #
    # for i, r in enumerate(rows):
    #     for j, c in enumerate(r):
    #         matrix[i*img_height:(i+1)*img_height,
    #                j*img_width:(j+1)*img_width] = c
    #

    matrix = np.vstack([np.hstack(r) for r in rows])

    c, r, _ = matrix.shape

    matrix = cv2.resize(matrix, (r * scale_factor, c * scale_factor))

    cv2.imshow(window_name, matrix)


# def display(window, img):
#     cv2.imshow(window, cv2.resize(
#         img, (img_size * scale_factor, img_size * scale_factor)))


@dataclass
class DistanceMetricConfig:
    norm_by_descr_dim: bool
    norm_multiplier: float | None
    metric: str


@dataclass
class SamplingConfig:
    different_objects: bool = False
    contrast_set_fraction: float = 0.5
    first_image_only: bool = False


@dataclass
class DisplayConfig:
    blend_weight_original_image: float = 0.3
    scale_factor: int = 2
    norm_diff_threshold: float = 0.05
    taper: float | int = 4
    normalize: bool = False

    window_name: str = "correspondence finder"


@dataclass
class HeatmapVisualizationConfig:
    distance: tuple[DistanceMetricConfig, ...]
    sampling: SamplingConfig
    display: DisplayConfig


class HeatmapVisualization:
    """
    Launches a live interactive heatmap visualization.

    Keypresses:
        n: new set of images
        s: swap images
        p: pause/un-pause
    """

    def __init__(
        self,
        replay_memory: DenseCorrespondenceDataset,
        encoders: tuple[KeypointsPredictor | VitFeatureEncoder, ...],
        config: HeatmapVisualizationConfig,
        cams: tuple[str, ...],
    ) -> None:
        self._config = config
        self.replay_memory = replay_memory
        self.encoders = encoders
        self.cams = cams

        self._paused = False

        self.image_height = replay_memory.scene_data.image_height
        self.image_width = replay_memory.scene_data.image_width

    def _get_new_images(self, cross_scene=True):
        cam_a = random.choice(self.cams)
        cam_b = random.choice(self.cams)
        obs_a, obs_b = self.replay_memory.sample_data_pair(
            cross_scene=cross_scene,
            cam_a=cam_a,
            cam_b=cam_b,
            contrast_obj=self._config.sampling.different_objects,
            contrast_fraction=self._config.sampling.contrast_set_fraction,
            first_image_only=self._config.sampling.first_image_only,
        )

        image_a_rgb = obs_a.rgb
        image_b_rgb = obs_b.rgb

        self.tensor1 = image_a_rgb
        self.tensor2 = image_b_rgb

        self.img1 = cv2.cvtColor(
            channel_front2back(image_a_rgb * 255).numpy().astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        self.img2 = cv2.cvtColor(
            channel_front2back(image_b_rgb * 255).numpy().astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )

    def _get_new_second_image(self, cross_scene=True):
        cam_b = random.choice(self.cams)
        _, obs_b = self.replay_memory.sample_data_pair(
            cross_scene=cross_scene,
            cam_a=cam_b,
            cam_b=cam_b,
            contrast_obj=self._config.sampling.different_objects,
            contrast_fraction=self._config.sampling.contrast_set_fraction,
            first_image_only=self._config.sampling.first_image_only,
        )

        image_b_rgb = obs_b.rgb

        self.tensor2 = image_b_rgb

        self.img2 = cv2.cvtColor(
            channel_front2back(image_b_rgb * 255).numpy().astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )

    def _get_descriptors(self, tensor) -> tuple[np.ndarray, ...]:
        return tuple(
            channel_front2back(
                e.compute_descriptor(tensor).detach().squeeze(0).cpu()
            ).numpy()
            for e in self.encoders
        )

    def _compute_descriptors(self):
        """
        Computes the descriptors for image 1 and image 2 for each network
        :return:
        :rtype:
        """
        self.img1_gray = cv2.cvtColor(self.img1, cv2.COLOR_RGB2GRAY) / 255.0
        self.img2_gray = cv2.cvtColor(self.img2, cv2.COLOR_RGB2GRAY) / 255.0

        self.res1 = self._get_descriptors(self.tensor1)
        self.res2 = self._get_descriptors(self.tensor2)

        self.find_best_match(None, 0, 0, None, None)

    def find_best_match(self, event, u, v, flags, param):
        """
        For each network, find the best match in the target image to point
        highlighted with reticle in the source image. Displays the result.
        :return:
        :rtype:
        """

        if self._paused:
            return

        scale_factor = self._config.display.scale_factor

        u = int(u / scale_factor) % self.image_width
        v = int(v / scale_factor) % self.image_height
        # print("Pos a", u, v)

        img_1_with_reticle = np.copy(self.img1)
        draw_reticle(img_1_with_reticle, u, v, COLOR_GREEN)

        alpha = self._config.display.blend_weight_original_image
        beta = 1 - alpha

        heatmaps_1 = []
        heatmaps_2 = []
        best_matches = []
        best_match_diffs = []
        best_matches_diffs_normed = []
        reticle_colors = []

        for dist_conf, res1, res2 in zip(self._config.distance, self.res1, self.res2):
            (
                best_match_uv,
                best_match_diff,
                norm_diffs_2,
            ) = correspondence_finder.find_best_match(
                (u, v), res1, res2, dist_conf.metric
            )

            best_matches.append(best_match_uv)
            best_match_diffs.append(best_match_diff)

            D = res1.shape[-1]

            norm_diffs_1 = correspondence_finder.get_norm_diffs(
                (u, v), res1, dist_conf.metric
            )

            # print("Pos b", best_match_uv)
            if dist_conf.norm_by_descr_dim:
                norm_diffs_2 = norm_diffs_2 / np.sqrt(D) * dist_conf.norm_multiplier
                norm_diffs_1 = norm_diffs_1 / np.sqrt(D) * dist_conf.norm_multiplier

            best_matches_diffs_normed.append(
                norm_diffs_2[best_match_uv[1], best_match_uv[0]]
            )

            threshold = self._config.display.norm_diff_threshold

            heatmap_color_1 = compute_gaussian_kernel_heatmap_from_norm_diffs(
                norm_diffs_1, self._config.display.taper, self._config.display.normalize
            )

            heatmap_color_2 = compute_gaussian_kernel_heatmap_from_norm_diffs(
                norm_diffs_2, self._config.display.taper, self._config.display.normalize
            )

            reticle_color = COLOR_RED if best_match_diff < threshold else COLOR_BLACK

            reticle_colors.append(reticle_color)

            draw_reticle(heatmap_color_1, u, v, reticle_color)

            draw_reticle(
                heatmap_color_2, best_match_uv[0], best_match_uv[1], reticle_color
            )

            blended_1 = cv2.addWeighted(self.img1, alpha, heatmap_color_1, beta, 0)
            blended_2 = cv2.addWeighted(self.img2, alpha, heatmap_color_2, beta, 0)

            # embed_1 = scale(res1, out_range=(0, 255)).astype(np.uint8)
            # embed_1 = embed_1[:, :, :3]  # for high dim descriptor, drop later dims
            # embed_1 = np.ascontiguousarray(embed_1, dtype=np.uint8)
            # draw_reticle(embed_1, u, v, COLOR_GREEN)

            heatmaps_1.append(blended_1)

            # embed_2 = scale(res2, out_range=(0, 255)).astype(np.uint8)
            # embed_2 = embed_2[:, :, :3]  # for high dim descriptor, drop later dims
            # embed_2 = np.ascontiguousarray(embed_2, dtype=np.uint8)
            # draw_reticle(embed_2, best_match_uv[0],
            #             best_match_uv[1], reticle_color)

            heatmaps_2.append(blended_2)

        img_2_with_reticle = np.copy(self.img2)

        for uv, color in zip(best_matches, reticle_colors):
            draw_reticle(img_2_with_reticle, uv[0], uv[1], color)

        display(
            (img_1_with_reticle, *heatmaps_1),
            (img_2_with_reticle, *heatmaps_2),
            scale_factor=scale_factor,
            window_name=self._config.display.window_name,
        )

        if event == cv2.EVENT_LBUTTONDOWN:
            info_dict = {
                "uv_a": (u, v),
                "uv_b": best_matches,
                "norm dist": best_match_diffs,
                "norm dist after normed": best_matches_diffs_normed,
            }
            print(info_dict)
            np.save("lid_header_norm_diffs.npy", norm_diffs_2)
            np.save("lid_header_rgb.npy", self.img2)

            # self.axes.hist(norm_diffs.flatten(), bins=50)
            # self.fig.savefig("hist_bef.png")
            # self.fig.canvas.draw()
            # self.fig.savefig("hist_aft.png")

    def run(self):
        # plt.ion()
        # self.fig = plt.figure()
        # self.axes = self.fig.add_subplot(111)

        window_name = self._config.display.window_name
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.find_best_match)

        self._get_new_images()
        self._compute_descriptors()

        while True:
            k = cv2.waitKey(40) & 0xFF
            if k == 27:
                break
            elif k == ord("n"):
                self._get_new_images()
                self._compute_descriptors()
            elif k == ord("r"):
                self._get_new_second_image()
                self._compute_descriptors()
            elif k == ord("s"):
                self.tensor1, self.tensor2 = self.tensor2, self.tensor1
                self.img1, self.img2 = self.img2, self.img1
                self._compute_descriptors()
            elif k == ord("p"):
                if self._paused:
                    self._paused = False
                else:
                    self._paused = True

        cv2.destroyAllWindows()
