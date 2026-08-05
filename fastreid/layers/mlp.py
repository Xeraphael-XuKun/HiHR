import torch
import torch.nn as nn
from typing import Iterable, List, Optional, Union
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Flexible MLP builder.

    Example:
      mlp = MLP(dims=[768, 512, 256, 10], activation="gelu", dropout=0.1)
      # layers: 768->512->256->10  (3个Linear, 2个隐藏层)

    Args:
      dims: e.g. [in_dim, h1, h2, ..., out_dim], len(dims) >= 2
      activation: "relu" | "gelu" | "tanh" | "silu" | "leaky_relu" | nn.Module
      dropout: float, applied after activation in hidden layers
      norm: None | "batchnorm" | "layernorm" (applied on hidden layers by default)
      final_activation: whether to apply activation (and norm/dropout) after last Linear
      bias: whether Linear has bias
    """

    def __init__(
            self,
            dims: Union[List[int], Iterable[int]],
            activation: Union[str, nn.Module] = "gelu",
            dropout: float = 0.0,
            norm: Optional[str] = None,
            final_activation: bool = False,
            bias: bool = True,
            leaky_relu_slope: float = 0.01,
            init: Union[str, nn.Module] = "auto",  # 新增：auto / kaiming / xavier / transformer
            last_layer_scale: float = 1.0  # 新增：最后一层缩放（想小一点就设 0.1 或 0.01）
    ):
        super().__init__()
        dims = list(dims)
        if len(dims) < 2:
            raise ValueError(f"`dims` must have at least 2 elements, got {dims}")

        self.activation_name = activation if isinstance(activation, str) else None
        self.leaky_relu_slope = leaky_relu_slope
        self.init = init
        self.last_layer_scale = last_layer_scale

        act = self._make_activation(activation, leaky_relu_slope)

        layers: List[nn.Module] = []
        num_linears = len(dims) - 1

        for i in range(num_linears):
            in_dim, out_dim = dims[i], dims[i + 1]
            layers.append(nn.Linear(in_dim, out_dim, bias=bias))

            is_last = (i == num_linears - 1)
            if is_last and not final_activation:
                continue  # 最后一层默认不加激活/norm/dropout

            # hidden (or final if final_activation=True)
            if norm is not None:
                n = norm.lower()
                if n in ("bn", "batchnorm", "batch_norm"):
                    layers.append(nn.BatchNorm1d(out_dim))
                elif n in ("ln", "layernorm", "layer_norm"):
                    layers.append(nn.LayerNorm(out_dim))
                else:
                    raise ValueError(f"Unknown norm: {norm}")

            layers.append(act)

            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

        self.reset_parameters()  # 初始化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward_residual(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.last_layer_scale * self.net(x)
        return F.normalize(x, dim=-1)

    def reset_parameters(self):
        act = (self.activation_name or "").lower()

        def init_linear(m: nn.Linear, is_last_linear: bool):
            if self.init == "transformer":
                # Transformer 常见：N(0, 0.02)
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            else:
                # auto：按激活选（你也可以强制 kaiming/xavier）
                if self.init in ("kaiming", "auto") and act in ("relu", "leaky_relu", "silu", "swish", "gelu", ""):
                    if act == "leaky_relu":
                        nn.init.kaiming_normal_(m.weight, a=self.leaky_relu_slope, mode="fan_in", nonlinearity="leaky_relu")
                    else:
                        nn.init.kaiming_normal_(m.weight, a=0.0, mode="fan_in", nonlinearity="relu")
                else:
                    gain = nn.init.calculate_gain("tanh") if act == "tanh" else 1.0
                    nn.init.xavier_uniform_(m.weight, gain=gain)

            if m.bias is not None:
                nn.init.zeros_(m.bias)

            if is_last_linear and self.last_layer_scale != 1.0:
                with torch.no_grad():
                    m.weight.mul_(self.last_layer_scale)
                    if m.bias is not None:
                        m.bias.mul_(self.last_layer_scale)

        # 找出 net 里所有 Linear，最后一个做 last_layer_scale
        linears = [m for m in self.modules() if isinstance(m, nn.Linear)]
        last_linear = linears[-1] if linears else None

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_linear(m, is_last_linear=(m is last_linear))
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _make_activation(activation: Union[str, nn.Module], leaky_relu_slope: float) -> nn.Module:
        if isinstance(activation, nn.Module):
            return activation

        a = activation.lower()
        if a == "relu":
            return nn.ReLU(inplace=True)
        if a == "gelu":
            return nn.GELU()
        if a in ("silu", "swish"):
            return nn.SiLU(inplace=True)
        if a == "tanh":
            return nn.Tanh()
        if a == "leaky_relu":
            return nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True)

        raise ValueError(f"Unknown activation: {activation}")
