import pathlib

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

from tapas_gmm.dataset.demos import Demos
from tapas_gmm.utils.franka import compute_ee_delta
from tapas_gmm.utils.geometry_np import (
    ensure_quat_positive_real_part,
    ensure_quaternion_continuity,
)
from tapas_gmm.utils.observation import (
    CameraOrder,
    SceneObservation,
    SingleCamObservation,
    dict_to_tensordict,
    empty_batchsize,
)


class Imitation_Learning_Dataset(Dataset):
    def __init__(
        self,
        data_folder: pathlib.Path,
        camera_names: tuple[str, ...] = tuple(("wrist", "overhead")),
    ):
        # self.wrist_camera_image = torch.from_numpy(
        #     np.load(data_folder / "wrist_camera_image.npy")
        # )
        # self.top_camera_image = torch.from_numpy(
        #     np.load(data_folder / "top_camera_image.npy")
        # )
        self.joint_states = torch.from_numpy(np.load(data_folder / "robot_dof_pos.npy"))
        # self.occupancy_map = torch.from_numpy(
        #     np.load(data_folder / "occupancy_map.npy")
        # )

        ee_poses = np.load(data_folder / "eetip_pose_global.npy")
        ee_poses[:, 3:] = ensure_quaternion_continuity(
            ensure_quat_positive_real_part(ee_poses[:, 3:])
        )
        self.ee_pose = torch.from_numpy(ee_poses)

        # self.next_ee_pose = torch.from_numpy(np.load(data_folder / "next_ee_pose.npy"))
        # self.ee_delta = compute_ee_delta(self.ee_pose, self.next_ee_pose)

        ee_pos_vel = torch.from_numpy(
            np.load(data_folder / "ee_velocities_reference_frame_pos.npy")
        )
        ee_ang_vel = torch.from_numpy(
            np.load(data_folder / "ee_velocities_reference_frame_rotvec.npy")
        )
        logger.info("Not taking proper EE rotatation deltas, but angular velocities!")
        self.ee_delta = torch.cat((ee_pos_vel, ee_ang_vel), dim=1)

        # self.gripper_width = torch.mean(self.joint_states[:, 23:24], dim=1).unsqueeze(1)
        self.gripper_width = (
            torch.from_numpy(np.load(data_folder / "do_grasp.npy")).float().unsqueeze(1)
        )

        # object_pose = torch.from_numpy(np.load(data_folder / "target_object_pose.npy"))
        self.object_poses = {}  # "obj": object_pose}

        self.camera_names = camera_names

    def __len__(self):
        return len(self.joint_states)

    def __getitem__(self, idx):
        sample = {
            # "wrist_camera_image": torch.from_numpy(self.wrist_camera_image[idx]),
            # "top_camera_image": torch.from_numpy(self.top_camera_image[idx]),
            # "robot_state": torch.from_numpy(self.robot_state[idx]),
            # "occupancy_map": torch.from_numpy(self.occupancy_map[idx]),
            "ee_pose": torch.from_numpy(self.ee_pose[idx]),
            "object_pose": torch.from_numpy(self.object_poses["obj"][idx]),
            # "next_ee_pose": torch.from_numpy(self.next_ee_pose[idx]),
        }
        return sample

    def to_tensor_dict(self) -> SceneObservation:
        # wrist_obs = SingleCamObservation(
        #     **{
        #         "rgb": self.wrist_camera_image,
        #         "depth": torch.zeros_like(self.wrist_camera_image),
        #         "extr": None,
        #         "intr": None,
        #     },
        #     batch_size=empty_batchsize,
        # )

        # overhead_obs = SingleCamObservation(
        #     **{
        #         "rgb": self.top_camera_image,
        #         "depth": torch.zeros_like(self.wrist_camera_image),
        #         "extr": None,
        #         "intr": None,
        #     },
        #     batch_size=empty_batchsize,
        # )

        # camera_obs = {"wrist": wrist_obs, "overhead": overhead_obs}

        # multicam_obs = dict_to_tensordict(
        #     {"_order": CameraOrder._create(self.camera_names)} | camera_obs
        # )

        scene_obs = SceneObservation(
            # cameras=multicam_obs,
            ee_pose=self.ee_pose,
            action=torch.cat((self.ee_delta, self.gripper_width), dim=1),
            object_poses=self.object_poses,
            joint_pos=None,
            joint_vel=None,
            gripper_state=self.gripper_width,
            batch_size=empty_batchsize,
        )

        return scene_obs


def make_demos(path: pathlib.Path) -> Demos:
    traj_dirs = [p for p in path.iterdir() if p.is_dir()]

    traj_data = [Imitation_Learning_Dataset(p) for p in traj_dirs]

    idcs = [i for i, t in enumerate(traj_data) if t.ee_pose[0, 4] > 0]
    traj_data = [traj_data[i] for i in idcs]

    ee_poses = [d.ee_pose for d in traj_data]
    # object_poses = [d.object_poses["obj"] for d in traj_data]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    for t in range(len(traj_data)):
        ax[0, 0].plot(ee_poses[t][:, 0], color="r")
        ax[0, 0].plot(ee_poses[t][:, 1], color="g")
        ax[0, 0].plot(ee_poses[t][:, 2], color="b")
        ax[0, 1].plot(ee_poses[t][:, 3], color="r")
        ax[0, 1].plot(ee_poses[t][:, 4], color="g")
        ax[0, 1].plot(ee_poses[t][:, 5], color="b")
        ax[0, 1].plot(ee_poses[t][:, 6], color="y")
        # ax[1, 0].plot(object_poses[t][:, 0], color="r")
        # ax[1, 0].plot(object_poses[t][:, 1], color="g")
        # ax[1, 0].plot(object_poses[t][:, 2], color="b")
        # ax[1, 1].plot(object_poses[t][:, 3], color="r")
        # ax[1, 1].plot(object_poses[t][:, 4], color="g")
        # ax[1, 1].plot(object_poses[t][:, 5], color="b")
        # ax[1, 1].plot(object_poses[t][:, 6], color="y")
    plt.show()

    # n_states = traj_data[0].robot_state.shape[1]
    # fig, ax = plt.subplots(n_states, 1, figsize=(5, n_states))
    # for i in range(n_states):
    #     ax[i].set_title(f"state {i}")
    #     for d in traj_data:
    #         ax[i].plot(d.robot_state[:, i])
    # plt.show()

    tensordicts = [d.to_tensor_dict() for d in traj_data]

    demo_meta_data = {
        "path": path,
    }

    data_kwargs = dict(
        meta_data=demo_meta_data,
        add_init_ee_pose_as_frame=True,
        add_world_frame=False,
        frames_from_keypoints=False,
        kp_indeces=None,
        enforce_z_down=False,
        enforce_z_up=False,
        modulo_object_z_rotation=False,
        make_quats_continuous=True,
    )

    demos = Demos(trajectories=tensordicts, **data_kwargs)

    return demos
