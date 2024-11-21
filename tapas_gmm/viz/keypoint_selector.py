from dataclasses import dataclass

import cv2

# import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger

import tapas_gmm.dense_correspondence.correspondence_finder as correspondence_finder
from tapas_gmm.viz.live_heatmap_visualization import (
    COLOR_BLACK,
    COLOR_GREEN,
    COLOR_RED,
    DistanceMetricConfig,
    compute_gaussian_kernel_heatmap_from_norm_diffs,
    display,
    draw_reticle,
)
from tapas_gmm.viz.operations import channel_front2back, channel_front2back_batch, scale


@dataclass
class DisplayConfig:
    blend_weight_original_image: float = 0.3
    scale_factor: int = 2
    norm_diff_threshold: float = 0.05
    taper: float | int = 4

    selection_window_name: str = "correspondence finder"
    preview_window_name: str = "trajectory preview"


@dataclass
class ManualKeypointSelectorConfig:
    distance: DistanceMetricConfig
    display: DisplayConfig


class KeypointSelector:
    def __init__(
        self,
        rgb: torch.Tensor,
        descriptor: torch.Tensor,
        mask: torch.Tensor,
        n_keypoints: int,
        config: ManualKeypointSelectorConfig,
        preview_rgb: torch.Tensor | None = None,
        preview_descr: torch.Tensor | None = None,
    ):
        self._config = config

        self.image_height, self.image_width = rgb.shape[:2]

        self.D = descriptor.shape[1]
        logger.info("Descriptor dim {}, only displaying first 3 dims.", self.D)

        self.rgb = cv2.cvtColor((rgb * 255).numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
        self.descriptor = channel_front2back(descriptor.squeeze(0).cpu()).numpy()
        self.descriptor_display = np.ascontiguousarray(
            scale(self.descriptor, out_range=(0, 255)).astype(np.uint8)[:, :, :3]
        )
        self.mask = (
            None
            if mask is None
            else cv2.cvtColor(mask.cpu().numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)
        )

        self.n_keypoints = n_keypoints

        self.selected_uv = []

        if preview_rgb is not None:
            self.preview_rgb = (
                channel_front2back_batch(preview_rgb).numpy().astype(np.uint8)
            )
            self.preview_descr = np.ascontiguousarray(
                channel_front2back_batch(preview_descr.cpu()).numpy()
            )
        else:
            self.preview_rgb = None

    def mark_location(self, event, u, v, flags, param):
        scale_factor = self._config.display.scale_factor

        u = int(u / scale_factor) % self.image_width
        v = int(v / scale_factor) % self.image_height

        rgb_with_reticle = np.copy(self.rgb)
        draw_reticle(rgb_with_reticle, u, v, COLOR_RED)

        descriptor_with_reticle = np.copy(self.descriptor_display)
        draw_reticle(descriptor_with_reticle, u, v, COLOR_RED)

        norm_diffs_1 = correspondence_finder.get_norm_diffs(
            (u, v), self.descriptor, metric=self._config.distance.metric
        )
        norm_diffs_1 /= np.sqrt(self.D)

        heatmap_color_1 = compute_gaussian_kernel_heatmap_from_norm_diffs(
            norm_diffs_1, self._config.display.taper, self._config.distance.metric
        )

        draw_reticle(heatmap_color_1, u, v, COLOR_RED)

        if self.mask is not None:
            mask_with_reticle = np.copy(self.mask)
            draw_reticle(mask_with_reticle, u, v, COLOR_RED)
        else:
            mask_with_reticle = np.zeros_like(rgb_with_reticle)

        display(
            (
                rgb_with_reticle,
                heatmap_color_1,
                descriptor_with_reticle,
                mask_with_reticle,
            ),
            scale_factor=scale_factor,
            window_name=self._config.display.selection_window_name,
        )

        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_uv.append((u, v))

            draw_reticle(self.rgb, u, v, COLOR_GREEN)
            draw_reticle(self.descriptor_display, u, v, COLOR_GREEN)
            draw_reticle(self.mask, u, v, COLOR_GREEN)

            logger.info(
                "Added ({},{}). Selected {}/{} keypoints",
                u,
                v,
                len(self.selected_uv),
                self.n_keypoints,
            )

        if self.preview_rgb is not None:
            B = self.preview_rgb.shape[0]
            preview = np.empty((B, self.image_height, self.image_width, 3))

            ref_descriptor = self.descriptor[v, u]

            alpha = self._config.display.blend_weight_original_image
            beta = 1 - alpha

            for t in range(B):
                norm_diffs_preview, (
                    up,
                    vp,
                ) = correspondence_finder.get_norm_diffs_to_ref(
                    ref_descriptor,
                    self.preview_descr[t],
                    metric=self._config.distance.metric,
                )
                norm_diffs_preview /= np.sqrt(self.D)

                heatmap = compute_gaussian_kernel_heatmap_from_norm_diffs(
                    norm_diffs_preview,
                    self._config.display.taper,
                    self._config.distance.metric,
                )

                img = cv2.cvtColor(self.preview_rgb[t], cv2.COLOR_RGB2BGR) / 255

                preview[t] = cv2.addWeighted(img, alpha, heatmap / 255, beta, 0)

                draw_reticle(preview[t], up, vp, COLOR_RED)

            preview = preview.reshape((4, 5, self.image_height, self.image_width, 3))

            display(
                *preview,
                scale_factor=1,
                window_name=self._config.display.preview_window_name
            )

    def run(self):
        # plt.ion()
        # self.fig = plt.figure()
        # self.axes = self.fig.add_subplot(111)

        # NOTE hat opencv 4.3.0 pip-installed. Remember there was some issue otherwise

        cv2.namedWindow(self._config.display.preview_window_name)

        cv2.namedWindow(self._config.display.selection_window_name)
        cv2.setMouseCallback(
            self._config.display.selection_window_name, self.mark_location
        )

        self.mark_location(None, 0, 0, None, None)

        while True:
            k = cv2.waitKey(40) & 0xFF
            if k == 27:
                break
            if len(self.selected_uv) == self.n_keypoints:
                cv2.destroyAllWindows()
                return self.selected_uv

        cv2.destroyAllWindows()
        return None
