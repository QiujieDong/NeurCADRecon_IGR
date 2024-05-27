import numpy as np
import torch.nn as nn
import torch
from torch.autograd import grad
from .embedder import get_embedder

def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0][:, -3:]
    return points_grad


# class ImplicitNet(nn.Module):
#     def __init__(
#         self,
#         d_in,
#         dims,
#         skip_in=(),
#         geometric_init=True,
#         radius_init=1,
#         beta=100
#     ):
#         super().__init__()
#
#         dims = [d_in] + dims + [1]
#
#         self.num_layers = len(dims)
#         self.skip_in = skip_in
#
#         for layer in range(0, self.num_layers - 1):
#
#             if layer + 1 in skip_in:
#                 out_dim = dims[layer + 1] - d_in
#             else:
#                 out_dim = dims[layer + 1]
#
#             lin = nn.Linear(dims[layer], out_dim)
#
#             # if true preform preform geometric initialization
#             if geometric_init:
#
#                 if layer == self.num_layers - 2:
#
#                     torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
#                     torch.nn.init.constant_(lin.bias, -radius_init)
#                 else:
#                     torch.nn.init.constant_(lin.bias, 0.0)
#
#                     torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
#
#             setattr(self, "lin" + str(layer), lin)
#
#         if beta > 0:
#             self.activation = nn.Softplus(beta=beta)
#
#         # vanilla relu
#         else:
#             self.activation = nn.ReLU()
#
#     def forward(self, input):
#
#         x = input
#
#         for layer in range(0, self.num_layers - 1):
#
#             lin = getattr(self, "lin" + str(layer))
#
#             if layer in self.skip_in:
#                 x = torch.cat([x, input], -1) / np.sqrt(2)
#
#             x = lin(x)
#
#             if layer < self.num_layers - 2:
#                 x = self.activation(x)
#
#         return x


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class ImplicitNet(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 n_layers,
                 skip_in=(),
                 d_hidden=256,
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(ImplicitNet, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        # return x / self.scale
        # sdf = torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)[:, :1]
        sdf = x[:, :1] / self.scale
        return sdf
