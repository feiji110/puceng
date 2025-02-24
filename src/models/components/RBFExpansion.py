#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   RBFExpansion.py
@Time    :   2023/12/14 00:22:40
@Author  :   Hengda.Gao
@Contact :   ghd@nudt.edu.com
'''
import torch
from torch import nn
from typing import Optional

class RBFExpansion(nn.Module):

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
        type: str = "gaussian"
    ) -> None:
        """Initialize a `RBFExpansion` module.

        :param vmin: The minimum value of the input.
        :param vmax: The maximum value of the input.
        :param bins: The number of bins to use.
        :param lengthscale: The lengthscale of the RBF kernel.
        :param type: The type of RBF kernel to use.
        """
        super().__init__()

        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(vmin, vmax, bins)
        )

        self.type = type


        if lengthscale is not None:
            self.lengthscale = lengthscale
            self.gamma = 1.0 / (lengthscale **2)
        else:
            self.lengthscale = torch.diff(self.centers).mean()
            self.gamma = 1.0 /self.lengthscale

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        base = self.gamma * (distance - self.centers)
        switcher = {
            'gaussian': (-base ** 2).exp(),
            'quadratic': base ** 2,
            'linear': base,
            'inverse_quadratic': 1.0 / (1.0 + base ** 2),
            'multiquadric': (1.0 + base ** 2).sqrt(),
            'inverse_multiquadric': 1.0 / (1.0 + base ** 2).sqrt(),
            'spline': base ** 2 * (base + 1.0).log(),
            'poisson_one': (base - 1.0) * (-base).exp(),
            'poisson_two': (base - 2.0) / 2.0 * base * (-base).exp(),
            'matern32': (1.0 + 3 ** 0.5 * base) * (-3 ** 0.5 * base).exp(),
            'matern52': (1.0 + 5 ** 0.5 * base + 5 / 3 * base ** 2) * (-5 ** 0.5 * base).exp(),
        }
        result = switcher.get(self.type, None)

        return result
        # if result.any():
        #     return result
        # else:
        #     raise Exception("No Implemented Radial Basis Method")

if __name__ == '__main__':
    a = RBFExpansion(lengthscale=8)
    print(a.centers)
    print(a.forward(  torch.ones(40) ))
