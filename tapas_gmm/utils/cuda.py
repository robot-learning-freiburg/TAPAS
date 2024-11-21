import torch
from loguru import logger

from tapas_gmm.utils.select_gpu import device

try:
    import pycuda.driver as cuda  # type: ignore
    from pycuda.compiler import SourceModule  # type: ignore

    FUSION_GPU_MODE = 1
except Exception as err:
    logger.warning("{}", err)
    logger.warning("Failed to import PyCUDA. Running fusion in CPU mode.")
    FUSION_GPU_MODE = 0
    cuda = None

kilo = 1024


def try_empty_cuda_cache():
    if device.type == "cpu":
        return
    try:
        empty_cuda_cache()
    except NameError:
        logger.warning("Could not empty cuda cache as CUDA seems not to run.")


def empty_cuda_cache():
    with torch.cuda.device(device):
        torch.cuda.empty_cache()


def make_context(device):
    gpu_num = device.index if device.type == "cuda" else None

    if gpu_num is None:
        return None
    else:
        assert cuda is not None
        return cuda.Device(gpu_num).make_context()


def try_make_context(device, dbg_memory=True):
    if FUSION_GPU_MODE:
        context = make_context(device)

        if dbg_memory:
            debug_memory(device)
    else:
        context = None

    return context


def destroy_context(context):
    if context is not None:
        context.pop()


def try_destroy_context(context):
    if FUSION_GPU_MODE:
        destroy_context(context)


def debug_memory(device):
    gpu_num = device.index if device.type == "cuda" else None

    if gpu_num is None:
        return None
    else:
        assert cuda is not None
        free, total = cuda.mem_get_info()
        free /= kilo**3
        total /= kilo**3
        logger.info(
            "{:.1f} GB of {:.1f} GB total memory are free on device {}",
            free,
            total,
            device,
        )


def try_debug_memory(device):
    if FUSION_GPU_MODE:
        debug_memory(device)
