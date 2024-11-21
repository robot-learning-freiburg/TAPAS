import os

import numpy as np
import torch
from loguru import logger


def get_gpu_with_most_free_mem(tmp_file="/tmp/gpu_mem"):
    os.system(f"nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >{tmp_file}")
    memory_available = [int(x.split()[2]) for x in open(tmp_file, "r").readlines()]
    os.system(f"rm {tmp_file}")
    return np.argmax(memory_available)


use_gpu = torch.cuda.is_available()
gpu_no = get_gpu_with_most_free_mem() if use_gpu else None
device = torch.device("cuda:{}".format(gpu_no) if use_gpu else "cpu")
logger.info("Running on {}", device)
