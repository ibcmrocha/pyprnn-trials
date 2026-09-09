#!/usr/bin/env python3
"""Vectorized plane-stress linear-elastic material model.

The elastic constants and Voigt convention intentionally match
``J2Tensor_vect.J2Material`` so both models return identical stresses while
the J2 model remains below its yield surface.
"""

import torch


class LinearElasticMaterial:
    """Stateless, vectorized isotropic plane-stress elasticity."""

    def __init__(self, device):
        self.E = 3.13e3
        self.nu_ = 0.37
        self.dev = device

    def configure(self, npoints):
        self.npoints = npoints
        self.el_Stiff = torch.zeros((npoints, 3, 3), device=self.dev)
        self.el_Stiff[:, 0, 0] = self.el_Stiff[:, 1, 1] = (
            self.E / (1.0 - self.nu_ * self.nu_)
        )
        self.el_Stiff[:, 0, 1] = self.el_Stiff[:, 1, 0] = (
            self.nu_ * self.E / (1.0 - self.nu_ * self.nu_)
        )
        self.el_Stiff[:, 2, 2] = 0.5 * self.E / (1.0 + self.nu_)

    def update(self, eps_new):
        strain = eps_new.view(self.npoints, 3, 1)
        return torch.bmm(self.el_Stiff, strain)[:, :, 0]

    def commit(self):
        """Match the stateful material-model interface; no state is stored."""

