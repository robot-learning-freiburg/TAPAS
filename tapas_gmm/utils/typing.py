import numpy as np
import torch

TensorOrTensorSeq = torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]
NDArrayOrNDArraySeq = np.ndarray | tuple[np.ndarray, ...] | list[np.ndarray]
