import math
import random
from argparse import ArgumentParser

import config as default_config
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from matplotlib import gridspec, patches
from sklearn import manifold

import tapas_gmm.utils.logging  # noqa
from tapas_gmm.dataset.dc import DenseCorrespondenceDataset
from tapas_gmm.encoder import encoder_switch
from tapas_gmm.utils.misc import (
    import_config_file,
    load_replay_memory,
    load_replay_memory_from_path,
    pretrain_checkpoint_name,
    set_seeds,
)
from tapas_gmm.utils.select_gpu import device
from tapas_gmm.viz.operations import channel_front2back_batch

RECONSTRUCTION = "reconstruction"
TSNE = "tsne"
SLOTS = "slots"


def vis_reconstruction(encoder, batch, config, batch_size):
    reconstruction = encoder.reconstruct(batch)
    reconstruction = reconstruction.cpu()

    no_cols = config["images_per_row"]
    no_rows = 2 * math.ceil(batch_size / no_cols)

    plt.figure(figsize=(no_cols + 1, no_rows + 1))

    gs = gridspec.GridSpec(
        no_rows,
        no_cols,
        wspace=0,
        hspace=0,
        top=1.0 - 0.5 / (no_rows + 1),
        bottom=0.5 / (no_rows + 1),
        left=0.5 / (no_cols + 1),
        right=1 - 0.5 / (no_cols + 1),
    )

    for i in range(batch_size):
        ax = plt.subplot(gs[2 * (i // no_cols), i % no_cols])
        ax.imshow(reconstruction[i].permute(1, 2, 0))
        ax.axis("off")
        ax.set_aspect("equal")
        ax = plt.subplot(gs[2 * (i // no_cols) + 1, i % no_cols])
        ax.imshow(batch[i].permute(1, 2, 0))
        ax.axis("off")
        ax.set_aspect("equal")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def vis_slots(encoder, batch, config, batch_size):
    reconstruction, _, _, _, _, _, slots = encoder.reconstruct_w_extras(batch)
    slots = slots.cpu()
    reconstruction = reconstruction.cpu()

    no_cols = len(config["encoder_config"]["slots"].keys()) + 1
    no_rows = batch_size

    plt.figure(figsize=(no_cols + 1, no_rows + 1))

    gs = gridspec.GridSpec(
        no_rows,
        no_cols,
        wspace=0,
        hspace=0,
        top=1.0 - 0.5 / (no_rows + 1),
        bottom=0.5 / (no_rows + 1),
        left=0.5 / (no_cols + 1),
        right=1 - 0.5 / (no_cols + 1),
    )

    for i in range(batch_size):
        ax = plt.subplot(gs[i, 0])
        ax.imshow(reconstruction[i].permute(1, 2, 0))
        ax.axis("off")
        ax.set_aspect("equal")
        for j in range(no_cols - 1):
            ax = plt.subplot(gs[i, j + 1])
            ax.imshow(slots[i][3 * j : 3 * (j + 1)].permute(1, 2, 0))
            ax.axis("off")
            ax.set_aspect("equal")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def vis_tsne(encoder, img, mask, config):
    embedding = channel_front2back_batch(
        encoder.compute_descriptor_batch(img).cpu().detach()
    ).numpy()

    n_points = config["tsne_no_points"]

    img_idx = random.choices(range(embedding.shape[0]), k=n_points)
    pos_x = random.choices(range(256), k=n_points)  # TODO: generic in img res
    pos_y = random.choices(range(256), k=n_points)

    data = np.stack([embedding[i, x, y] for i, x, y in zip(img_idx, pos_x, pos_y)])

    tsne = manifold.TSNE(n_components=2, init="pca", random_state=0)
    trans_data = tsne.fit_transform(data).T

    mask = mask.numpy()

    colors = [mask[i, x, y] for i, x, y in zip(img_idx, pos_x, pos_y)]

    plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)
    entries = [
        patches.Patch(color=plt.cm.rainbow(c), label=str(c)) for c in np.unique(colors)
    ]

    plt.legend(handles=entries)
    plt.show()


def main(config, path=None):
    Encoder = encoder_switch[config["encoder"]]
    encoder = Encoder(config["encoder_config"]).to(device)
    file_name = pretrain_checkpoint_name(config)
    encoder.from_disk(file_name)
    encoder.eval()
    batch_size = config["batch_size"]

    encoder_config = encoder_configs[config["policy_config"]["encoder"]]
    if path:
        replay_memory = load_replay_memory_from_path(path)
    else:
        replay_memory = load_replay_memory(config)
    replay_memory = DenseCorrespondenceDataset(replay_memory, None)  # encoder_config)
    with torch.no_grad():  # at some point switch to new inference_mode
        # TODO: only doing reconstruction of single image.
        # also test transport capabilities for tranporter!

        if args.visualization == RECONSTRUCTION:
            batch = replay_memory.variable_sample(
                batch_size, "camera_single", cam=config["cam"]
            )
            vis_reconstruction(encoder, batch, config, batch_size)
        elif args.visualization == SLOTS:
            batch = replay_memory.variable_sample(
                batch_size, "camera_single", cam=config["cam"]
            )
            assert config["encoder"] == "monet"
            vis_slots(encoder, batch, config, batch_size)
        elif args.visualization == TSNE:
            img, mask = replay_memory.sample_images_across_trajectories(
                batch_size, cam=config["cam"]
            )
            vis_tsne(encoder, img, mask, config)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--feedback_type",
        dest="feedback_type",
        default="cloning_10",
        help="options: cloning_10, cloning_100",
    )
    parser.add_argument(
        "--pretrain_feedback",
        dest="pretrain_feedback",
        default=None,
        help="The data on which the model was pretrained, eg. dcs_20. ",
    )
    parser.add_argument(
        "-t",
        "--task",
        dest="task",
        default="CloseMicrowave",
        # help="options: {}, 'Mixed'".format(str(tasks)[1:-1]),
    )
    parser.add_argument(
        "-e",
        "--encoder",
        dest="encoder",
        default="transporter",
        help="options: transporter, bvae, monet, keypoints",
    )
    parser.add_argument(
        "-v",
        "--visualization",
        dest="visualization",
        default=RECONSTRUCTION,
        help="options: {}, {}, {}".format(RECONSTRUCTION, TSNE, SLOTS),
    )
    parser.add_argument(
        "-m",
        "--mask",
        dest="mask",
        action="store_true",
        default=False,
        help="Use data with ground truth object masks.",
    )
    parser.add_argument(
        "-o",
        "--object_pose",
        dest="object_pose",
        action="store_true",
        default=False,
        help="Use data with ground truth object positions.",
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default=None,
        help="Config file to use. Uses default if None provided.",
    )
    parser.add_argument(
        "--encoder_suffix",
        dest="encoder_suffix",
        default=None,
        help="Pass a suffix to append to the name of the encoder checkpoint.",
    )
    parser.add_argument(
        "--cam",
        dest="cam",
        required=True,
        nargs="+",
        help="The camera(s) to use. Options: wrist, overhead.",
    )
    parser.add_argument(
        "--path",
        dest="path",
        default=None,
        help="Path to a dataset. May be provided instead of f-t-m.",
    )
    args = parser.parse_args()

    set_seeds()

    if args.config is not None:
        encoder_configs = import_config_file(args.config).encoder_configs
    else:
        encoder_configs = default_config.encoder_configs
    config_defaults = {
        "feedback_type": args.feedback_type,
        "task": args.task,
        "pretrain_feedback": args.pretrain_feedback,
        "encoder": args.encoder,
        "visualization": args.visualization,
        "batch_size": 4,
        "tsne_no_points": 500,
        "images_per_row": 8,  # for reconstruction. optimal: bs/ipr=ipr/2
        "encoder_config": encoder_configs[args.encoder],
        "ground_truth_mask": args.mask or args.object_pose,
        "ground_truth_object_pose": args.object_pose,
        "cam": args.cam,
        "data_root": "data",
        "policy_config": {
            "lstm_layers": 2,
            "n_cams": len(args.cam),
            "use_ee_pose": True,  # else uses joint angles
            "action_dim": 7,
            "learning_rate": 3e-4,
            "weight_decay": 3e-6,
            "encoder": args.encoder,
            "encoder_config": encoder_configs["keypoints"],
            "encoder_suffix": args.encoder_suffix,
        },
    }
    main(config_defaults, path=args.path)
