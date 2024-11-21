from enum import Enum

from tapas_gmm.policy.policy import Policy

# from loguru import logger

# from tapas_gmm.policy.encoder import DiskReadEncoderPolicy, EncoderPolicy
# from tapas_gmm.policy.gmm import GMMPolicy
# from tapas_gmm.policy.lstm import LSTMPolicy
# from tapas_gmm.policy.manual import ManualPolicy

# try:
#     from tapas_gmm.policy.motion_planner import MotionPlannerPolicy
# except ImportError:
#     logger.error("Can't import MotionPlannerPolicy. Is mplib installed?")
#     MotionPlannerPolicy = None

# from tapas_gmm.policy.diffusion import DiffusionPolicy
# from tapas_gmm.policy.random import RandomPolicy
# from tapas_gmm.policy.sphere import SpherePolicy

# # TODO: change this to auto-select the policy based on the policy config class?
# policy_switch = {
#     "encoder": EncoderPolicy,
#     "random": RandomPolicy,
#     "sphere": SpherePolicy,
#     "manual": ManualPolicy,
#     "motion_planner": MotionPlannerPolicy,
#     "gmm": GMMPolicy,
#     "lstm": LSTMPolicy,
#     "diffusion": DiffusionPolicy,
# }

# policy_names = list(policy_switch.keys())


class PolicyEnum(Enum):
    RANDOM = "random"
    MANUAL = "manual"
    SPHERE = "sphere"
    MOTION_PLANNER = "motion_planner"
    GMM = "gmm"
    LSTM = "lstm"
    DIFFUSION = "diffusion"


# def get_policy_class(policy_name, disk_read=False):
#     # if disk_read:
#     #     return DiskReadEncoderPolicy
#     # else:
#     #     return policy_switch[policy_name]
#     return policy_switch[policy_name]


# TODO: switch over to config-based policy import like for envs
def import_policy(policy_name: str, disk_read=False) -> Policy:
    if policy_name == "encoder":
        from tapas_gmm.policy.encoder import EncoderPolicy as Policy
    elif policy_name == "random":
        from tapas_gmm.policy.random import RandomPolicy as Policy
    elif policy_name == "sphere":
        from tapas_gmm.policy.sphere import SpherePolicy as Policy
    elif policy_name == "manual":
        from tapas_gmm.policy.manual import ManualPolicy as Policy
    elif policy_name == "motion_planner":
        from tapas_gmm.policy.motion_planner import MotionPlannerPolicy as Policy
    elif policy_name == "gmm":
        from tapas_gmm.policy.gmm import GMMPolicy as Policy
    elif policy_name == "lstm":
        from tapas_gmm.policy.lstm import LSTMPolicy as Policy
    elif policy_name == "diffusion":
        from tapas_gmm.policy.diffusion import DiffusionPolicy as Policy
    else:
        raise ValueError(f"Invalid policy {policy_name}")

    return Policy
