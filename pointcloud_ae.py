#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pointcloud_ae.py

PointNet-style AutoEncoder for 3D point clouds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointCloudAE(nn.Module):
    """
    PointNet-style AutoEncoder for 3D point clouds.

    Args:
        point_size (int): Number of points per cloud.
        latent_size (int): Dimension of latent embedding.
    """

    def __init__(self, point_size=1024, latent_size=512):
        super(PointCloudAE, self).__init__()
        self.latent_size = latent_size
        self.point_size = point_size

        # Encoder
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(256, 256, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

        self.conv4 = nn.Conv1d(256, 256, 1)
        self.conv5 = nn.Conv1d(256, 256, 1)
        self.conv6 = nn.Conv1d(256, latent_size, 1)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(latent_size)

        # Decoder
        self.dec1 = nn.Linear(latent_size, 256)
        self.dec2 = nn.Linear(256, 256)
        self.dec3 = nn.Linear(256, point_size * 3)

    # ---------------------------------------------------------
    def encoder(self, x):
        """
        Encode input point cloud x of shape (B, 3, N) to latent code.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x + torch.max(x, 2, keepdim=True)[0]  # skip connection

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.bn6(self.conv6(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.latent_size)
        return x

    def decoder(self, z):
        """
        Decode latent code z to reconstructed point cloud (B, N, 3).
        """
        z = F.relu(self.dec1(z))
        z = F.relu(self.dec2(z))
        z = self.dec3(z)
        return z.view(-1, self.point_size, 3)

    def forward(self, x):
        """
        Forward pass: encode then decode.
        """
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def chamfer_distance(pcd1: torch.Tensor, pcd2: torch.Tensor) -> torch.Tensor:
    """
    Compute Chamfer distance between two batches of point clouds.

    Args:
        pcd1: (B, N, d)
        pcd2: (B, M, d)

    Returns:
        Mean Chamfer distance scalar.
    """
    B, N, d = pcd1.shape
    _, M, _ = pcd2.shape
    diff = pcd1.unsqueeze(2) - pcd2.unsqueeze(1)
    dist_sq = torch.sum(diff ** 2, dim=-1)
    dist1, _ = torch.min(dist_sq, dim=2)
    dist2, _ = torch.min(dist_sq, dim=1)
    cd = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
    return torch.mean(cd)
