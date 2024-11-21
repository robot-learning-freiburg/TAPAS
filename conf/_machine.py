from omegaconf import MISSING

from tapas_gmm.utils.misc import DataNamingConfig

data_naming_config = DataNamingConfig(
    feedback_type=MISSING, task=MISSING, data_root="data"
)


CUDA_PATH = "/usr/local/cuda-11.1"
LD_LIBRARY_PATH = ":".join(
    [CUDA_PATH + "/lib64", "/home/hartzj/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04"]
)
