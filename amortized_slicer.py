#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
amortized_slicer_trainer.py

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


# ======================================================================
#                           CLASS DEFINITION
# ======================================================================

class AmortizedSlicerTrainer:
    """
    Wrapper for amortized min-STP training and evaluation.
    """

    def __init__(
        self,
        autoencoder,
        soft_sort,
        hard_sort,
        soft_permutation,
        device,
        pair,
        train_loader,
        test_loader,
        model_class,
        latent_dim=512,
        alpha_val=5.0,
        lr=1e-4,
        step_size=20,
        gamma=0.95,
        inner_epochs=100,
        outer_epochs=25,
        save_dir="./ckpts/",
        seed=None,
    ):
        self.device = device
        self.net = autoencoder.to(device)
        self.soft_sort = soft_sort
        self.hard_sort = hard_sort
        self.soft_permutation = soft_permutation
        self.pair = pair
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model_class = model_class
        self.latent_dim = latent_dim
        self.alpha = torch.tensor(alpha_val, requires_grad=False).to(device)
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.inner_epochs = inner_epochs
        self.outer_epochs = outer_epochs
        self.save_dir = save_dir

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        os.makedirs(save_dir, exist_ok=True)

    # --------------------------------------------------------------
    def _compute_context(self, pc):
        """Encode a point cloud batch using AE encoder."""
        pc = pc.to(self.device)
        x = pc.pos.view(pc.num_graphs, -1, 3).permute(0, -1, 1)
        c = self.net.encoder(x).detach().squeeze()
        return c, pc.pos.view(pc.num_graphs, -1, 3)[0]

    def _pair_forward(self, X, Y, cX, cY, model, context_proj):
        """Forward pass returning projected embeddings and distance matrix."""
        Nx, _ = X.shape
        Ny, _ = Y.shape
        M = torch.sqrt(ot.dist(X, Y))
        a = torch.ones(Nx, device=self.device) / Nx
        b = torch.ones(Ny, device=self.device) / Ny

        cx_proj = context_proj(cX)
        cy_proj = context_proj(cY)
        C = (cx_proj + cy_proj) / 2

        Xinput = torch.cat([X, C.repeat(Nx, 1)], dim=-1)
        Yinput = torch.cat([Y, C.repeat(Ny, 1)], dim=-1)

        Xout, Yout = model(Xinput).squeeze(), model(Yinput).squeeze()
        return Xout, Yout, M, a, b

    # --------------------------------------------------------------
    def train(self):
        """Main amortized min-STP training loop."""
        model = self.model_class(architecture=[6, 256, 512, 256, 1]).to(self.device)
        context_proj = nn.Linear(self.latent_dim, 3).to(self.device)
        optimizer = optim.AdamW(
            list(model.parameters()) + list(context_proj.parameters()), lr=self.lr
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )

        losses, test_costs, train_costs = [], [], []

        for e in range(self.outer_epochs):
            optimizer.zero_grad()
            for pc1, pc2 in tqdm(self.train_loader, desc=f"Epoch {e+1}/{self.outer_epochs}"):
                cX, X = self._compute_context(pc1)
                cY, Y = self._compute_context(pc2)
            
                for _ in range(self.inner_epochs):
                    model.train()
                    

                    Xout, Yout, M, _, _ = self._pair_forward(X, Y, cX, cY, model, context_proj)

                    # Two-pass symmetric soft/hard plan
                    Px = self.soft_permutation(Xout, alpha=self.alpha)
                    Px = torch.maximum(Px, torch.tensor(1e-6, device=self.device))
                    Py = self.hard_sort(Yout.unsqueeze(0)).squeeze()
                    P1 = Px.T @ Py

                    Px = self.hard_sort(Xout.unsqueeze(0)).squeeze()
                    Py = self.soft_permutation(Yout, alpha=self.alpha)
                    Py = torch.maximum(Py, torch.tensor(1e-6, device=self.device))
                    P2 = Px.T @ Py

                    pi = (P1 + P2) / 2
                    loss = (pi * M).sum()
                    loss.backward()
            
                    losses.append(loss.item())
            optimizer.step()
            scheduler.step()

            # Evaluation
            if e % 6 == 0 or e == self.outer_epochs - 1:
                print(f"\n--- Evaluation after epoch {e+1} ---")
                test_mean = self.evaluate(model, context_proj, self.test_loader)
                train_mean = self.evaluate(model, context_proj, self.train_loader)
                print(f"Train cost: {train_mean:.4f} | Test cost: {test_mean:.4f}")
                test_costs.append(test_mean)
                train_costs.append(train_mean)

        # Save
        self._save_costs(train_costs, test_costs)
        return model, context_proj, losses, train_costs, test_costs

    # --------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, model, context_proj, loader):
        """Compute 1D OT (emd_1d) cost per sample."""
        costs = []
        for pc1, pc2 in loader:
            cX, X = self._compute_context(pc1)
            cY, Y = self._compute_context(pc2)
            Xout, Yout, M, a, b = self._pair_forward(X, Y, cX, cY, model, context_proj)
            plans = ot.emd_1d(Xout, Yout, a, b)
            cost = (plans * M).sum()
            costs.append(cost.item())
        return float(np.mean(costs))

    def _save_costs(self, train_costs, test_costs):
        """Save computed costs."""
        pairname = f"{self.pair[0]}_{self.pair[1]}"
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, f"train_costs_{pairname}.pkl"), "wb") as f:
            pickle.dump(np.array(train_costs), f)
        with open(os.path.join(self.save_dir, f"test_costs_{pairname}.pkl"), "wb") as f:
            pickle.dump(np.array(test_costs), f)
        print(f"[✓] Saved cost arrays for {pairname} to {self.save_dir}")


# ======================================================================
#                            MAIN EXECUTION
# ======================================================================

if __name__ == "__main__":
    import sys
    from pointcloud_ae import PointCloudAE
    from data import PairedModelNet
    from models import MLPWithSkipConnections
    from softsort import SoftSort_p2 as SoftSort
    from lapsum import soft_permutation
    from torch_geometric.datasets import ModelNet
    from torch_geometric.loader import DataLoader
    import torch_geometric.transforms as T

    # ------------------------- CLI Arguments -------------------------
    parser = argparse.ArgumentParser(description="Amortized Min-STP Trainer")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (cpu | cuda | cuda:IDX)")
    parser.add_argument("--pair", nargs=2, default=["monitor", "desk"], metavar=("CLS1", "CLS2"))
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--outer_epochs", type=int, default=25, help="Number of outer epochs")
    parser.add_argument("--inner_epochs", type=int, default=100, help="Number of inner optimization steps per batch")
    parser.add_argument("--save_dir", type=str, default="./ckpts", help="Directory to save outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    pair = args.pair

    # ------------------------- Dataset Setup -------------------------
    transform = T.SamplePoints(1024)
    pre_transform = T.NormalizeScale()
    train_dataset = ModelNet("ModelNet10", name="10", train=True, transform=transform, pre_transform=pre_transform)
    test_dataset = ModelNet("ModelNet10", name="10", train=False, transform=transform, pre_transform=pre_transform)

    train_paired = PairedModelNet(train_dataset, pair[0], pair[1], random_pair=False)
    test_paired = PairedModelNet(test_dataset, pair[0], pair[1], random_pair=False)
    train_loader = DataLoader(train_paired, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_paired, batch_size=1, shuffle=False)

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

    # ------------------------- Trainer Init -------------------------
    trainer = AmortizedSlicerTrainer(
        autoencoder=net,
        soft_sort=soft_sort,
        hard_sort=hard_sort,
        soft_permutation=soft_permutation,
        device=device,
        pair=pair,
        train_loader=train_loader,
        test_loader=test_loader,
        model_class=MLPWithSkipConnections,
        lr=args.lr,
        outer_epochs=args.outer_epochs,
        inner_epochs=args.inner_epochs,
        save_dir=args.save_dir,
        seed=args.seed,
    )

    # ------------------------- Run Training -------------------------
    model, context_proj, losses, train_costs, test_costs = trainer.train()
