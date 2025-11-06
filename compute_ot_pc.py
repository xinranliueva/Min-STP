#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
amortized_slicer_trainer.py
Author: Xinran (Eva) Liu

Trainer for amortized min-STP learning on paired ModelNet10 classes.
Compatible with pointcloud_ae.py and data.py.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import ot

import gdown
import os
import sys
from pointcloud_ae import PointCloudAE
from data import PairedModelNet
from models import MLPWithSkipConnections
from softsort import SoftSort_p2 as SoftSort
from lapsum import soft_permutation
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

device = "cuda:1"

transform = T.SamplePoints(1024)
pre_transform = T.NormalizeScale()
train_dataset = ModelNet("ModelNet10", name="10", train=True, transform=transform, pre_transform=pre_transform)
test_dataset = ModelNet("ModelNet10", name="10", train=False, transform=transform, pre_transform=pre_transform)

pair = ["chair","table"]

# train_paired = PairedModelNet(train_dataset, pair[0], pair[1], random_pair=False)
# test_paired = PairedModelNet(test_dataset, pair[0], pair[1], random_pair=False)
# train_loader = DataLoader(train_paired, batch_size=1, shuffle=False)
# test_loader = DataLoader(test_paired, batch_size=1, shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ------------------------- Model Setup -------------------------
net = PointCloudAE(point_size=1024, latent_size=512)
ckpt_path = "./ModelNet10/model_checkpoint.pth"
if os.path.exists(ckpt_path):
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"[✓] Loaded AE checkpoint from {ckpt_path}")
else:
    print(f"[!] Warning: AE checkpoint not found at {ckpt_path}")

soft_sort = SoftSort(tau=1e-2, hard=False)
hard_sort = SoftSort(hard=True)


def compute_context(pc):
    """Encode a point cloud batch using AE encoder."""
    pc = pc.to(device)
    x = pc.pos.view(pc.num_graphs, -1, 3).permute(0, -1, 1)
    return pc.pos.view(pc.num_graphs, -1, 3)[0]


costs_ot= []
costs_rand= []
# for pc1, pc2 in test_loader:
for pc2 in test_loader:
    # X = compute_context(pc1)
    # Y = compute_context(pc2)
    Y = compute_context(pc2)
    X = torch.randn(Y.shape).to(Y.device)

    M = torch.sqrt(ot.dist(X, Y))
    Nx, _ = X.shape
    Ny, _ = Y.shape
    a = torch.ones(Nx, device=device) / Nx
    b = torch.ones(Ny, device=device) / Ny

    plans_ot = ot.emd(a, b, M)

    cost_ot = (plans_ot * M).sum()
    costs_ot.append(cost_ot.item())

    plans_rand = torch.zeros(X.shape[0], Y.shape[0], device=X.device)
    plans_rand[torch.arange(X.shape[0]), torch.randperm(Y.shape[0])] = 1 / X.shape[0]
    # print(plans_rand.sum(dim=0))
    # print(plans_rand.sum(dim=1))

    cost_rand = (plans_rand * M).sum()
    costs_rand.append(cost_rand.item())

print(float(np.mean(costs_ot)))
print(float(np.mean(costs_rand)))


