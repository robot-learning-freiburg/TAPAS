from abc import abstractmethod

from torch import nn

from tapas_gmm.encoder.models.bvae.types_ import Any, List, Tensor


class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        raise NotImplementedError
