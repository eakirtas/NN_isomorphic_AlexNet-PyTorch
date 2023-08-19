import torch as T
from torch.nn.common_types import _size_1_t

# from torch.nn.modules.typing import Union


class NNConv1d(T.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_1_t,
            w_pos: T.Tensor,
            w_neg: T.Tensor,
            bias: T.Tensor,
            alpha: T.Tensor,
            activation_shift: T.Tensor,
            stride: _size_1_t = 1,
            padding=0,  # Union[str, _size_1_t]
            dilation: _size_1_t = 1,
            groups: int = 1,
            padding_mode: str = 'zeros',  # TODO: refine this type
            device=None,
            dtype=None) -> None:
        super().__init__()
        super().cuda()

        self.l_pos = T.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            padding_mode=padding_mode,  # TODO: refine this type
            device=device,
            dtype=dtype)

        self.l_neg = T.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            padding_mode=padding_mode,  # TODO: refine this type
            device=device,
            dtype=dtype)

        self.bias = T.nn.Parameter(bias.cuda())

        self.l_pos.weight = T.nn.Parameter(w_pos.cuda())
        self.l_neg.weight = T.nn.Parameter(w_neg.cuda())

        self.register_buffer('alpha', alpha)

        self.register_buffer('activation_shift', activation_shift)

    def forward(self, x: T.Tensor) -> T.Tensor:
        x_t = self.alpha - x

        x_pos = self.l_pos(x)
        x_neg = self.l_neg(x_t)

        x = x_pos + x_neg + self.bias.view(1, -1, 1)

        x = x - self.activation_shift

        return x

    def get_weight(self):
        return self.l_pos.weight + self.l_neg.weight
