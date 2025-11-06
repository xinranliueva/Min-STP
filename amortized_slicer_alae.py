#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
amortized_slicer_trainer.py
Author: Xinran (Eva) Liu

Trainer for amortized min-STP learning on Image-to-Image translation.
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

# ======================================================================
#                           I2I DATASET
# ======================================================================

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.indices_x = torch.arange(x.shape[0])
        self.indices_y = torch.arange(y.shape[0])

    def __len__(self):
        return min(self.x.shape[0], self.y.shape[0])

    def __getitem__(self, index):
        index_x = self.indices_x[index]
        index_y = self.indices_y[index]
        return self.x[index_x], self.y[index_y]

    def shuffle(self):
        self.indices_x = torch.randperm(self.indices_x.shape[0])
        self.indices_y = torch.randperm(self.indices_y.shape[0])



def process_data(input_data = "ADULT", target_data = "CHILDREN"):
    if not os.path.isdir('./data'):
        os.makedirs('./data')

    urls = {
        "./data/age.npy": "https://drive.google.com/uc?id=1Vi6NzxCsS23GBNq48E-97Z9UuIuNaxPJ",
        "./data/gender.npy": "https://drive.google.com/uc?id=1SEdsmQGL3mOok1CPTBEfc_O1750fGRtf",
        "./data/latents.npy": "https://drive.google.com/uc?id=1ENhiTRsHtSjIjoRu1xYprcpNd8M9aVu8",
        "./data/test_images.npy": "https://drive.google.com/uc?id=1SjBWWlPjq-dxX4kxzW-Zn3iUR3po8Z0i",
    }

    for name, url in urls.items():
        gdown.download(url, os.path.join(f"{name}"), quiet=False)

    ## process data
    train_size = 60000
    test_size = 10000

    latents = np.load("./data/latents.npy")
    gender = np.load("./data/gender.npy")
    age = np.load("./data/age.npy")
    test_inp_images = np.load("./data/test_images.npy")

    train_latents, test_latents = latents[:train_size], latents[train_size:]
    train_gender, test_gender = gender[:train_size], gender[train_size:]
    train_age, test_age = age[:train_size], age[train_size:]

    if input_data == "MAN":
        x_inds_train = np.arange(train_size)[(train_gender == "male").reshape(-1)]
        x_inds_test = np.arange(test_size)[(test_gender == "male").reshape(-1)]
    elif input_data == "WOMAN":
        x_inds_train = np.arange(train_size)[(train_gender == "female").reshape(-1)]
        x_inds_test = np.arange(test_size)[(test_gender == "female").reshape(-1)]
    elif input_data == "ADULT":
        x_inds_train = np.arange(train_size)[
            (train_age >= 18).reshape(-1)*(train_age != -1).reshape(-1)
        ]
        x_inds_test = np.arange(test_size)[
            (test_age >= 18).reshape(-1)*(test_age != -1).reshape(-1)
        ]
    elif input_data == "CHILDREN":
        x_inds_train = np.arange(train_size)[
            (train_age < 18).reshape(-1)*(train_age != -1).reshape(-1)
        ]
        x_inds_test = np.arange(test_size)[
            (test_age < 18).reshape(-1)*(test_age != -1).reshape(-1)
        ]
    x_data_train = train_latents[x_inds_train]
    x_data_test = test_latents[x_inds_test]

    if target_data == "MAN":
        y_inds_train = np.arange(train_size)[(train_gender == "male").reshape(-1)]
        y_inds_test = np.arange(test_size)[(test_gender == "male").reshape(-1)]
    elif target_data == "WOMAN":
        y_inds_train = np.arange(train_size)[(train_gender == "female").reshape(-1)]
        y_inds_test = np.arange(test_size)[(test_gender == "female").reshape(-1)]
    elif target_data == "ADULT":
        y_inds_train = np.arange(train_size)[
            (train_age >= 18).reshape(-1)*(train_age != -1).reshape(-1)
        ]
        y_inds_test = np.arange(test_size)[
            (test_age >= 18).reshape(-1)*(test_age != -1).reshape(-1)
        ]
    elif target_data == "CHILDREN":
        y_inds_train = np.arange(train_size)[
            (train_age < 18).reshape(-1)*(train_age != -1).reshape(-1)
        ]
        y_inds_test = np.arange(test_size)[
            (test_age < 18).reshape(-1)*(test_age != -1).reshape(-1)
        ]
    y_data_train = train_latents[y_inds_train]
    y_data_test = test_latents[y_inds_test]

    X_train = torch.tensor(x_data_train)
    Y_train = torch.tensor(y_data_train)

    X_test = torch.tensor(x_data_test)
    Y_test = torch.tensor(y_data_test)

    return X_train, Y_train, X_test, Y_test


# ======================================================================
#                           CLASS DEFINITION
# ======================================================================

class AmortizedSlicerTrainer:
    """
    Wrapper for amortized min-STP training and evaluation.
    """

    def __init__(
        self,
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
        self.soft_sort = soft_sort
        self.hard_sort = hard_sort
        self.soft_permutation = soft_permutation
        self.pair = pair
        self.train_loader=train_loader
        self.test_loader=test_loader
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
    

    def _pair_forward_latents(self, X, Y, model):
        """Forward pass returning projected embeddings and distance matrix."""
        Nx, _ = X.shape
        Ny, _ = Y.shape
        M = torch.sqrt(ot.dist(X, Y))
        a = torch.ones(Nx, device=self.device) / Nx
        b = torch.ones(Ny, device=self.device) / Ny

        Xinput = X
        Yinput = Y

        Xout, Yout = model(Xinput).squeeze(), model(Yinput).squeeze()
        return Xout, Yout, M, a, b

    # --------------------------------------------------------------
    def train(self):
        """Main amortized min-STP training loop."""
        model = self.model_class(architecture=[512, 256, 512, 256, 1]).to(self.device)
        context_proj = nn.Linear(self.latent_dim, 3).to(self.device)
        optimizer = optim.AdamW(
            list(model.parameters()) + list(context_proj.parameters()), lr=self.lr
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )

        losses, test_costs, train_costs = [], [], []

        for e in range(self.outer_epochs):
            losses_outer = []
            for X, Y in tqdm(self.train_loader, desc=f"Epoch {e+1}/{self.outer_epochs}"):
            # for X, Y in self.train_loader:
                X = X.to(device)
                Y = Y.to(device)
                for _ in range(self.inner_epochs):
                    model.train()
                    optimizer.zero_grad()

                    Xout, Yout, M, _, _ = self._pair_forward_latents(X, Y, model)

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
                    optimizer.step()
                    losses.append(loss.item())

                    losses_outer.append(loss.item())

                scheduler.step()
            print("Loss: ", np.array(losses_outer).sum() / (len(self.train_loader) * self.inner_epochs))
            # Evaluation
            if e % 6 == 0 or e == self.outer_epochs - 1:
                print(f"\n--- Evaluation after epoch {e+1} ---")
                test_mean = self.evaluate(model, self.test_loader)
                train_mean = self.evaluate(model, self.train_loader)
                print(f"Train cost: {train_mean:.4f} | Test cost: {test_mean:.4f}")
                test_costs.append(test_mean)
                train_costs.append(train_mean)

                # Save
                self._save_costs(train_costs, test_costs)
                
                torch.save(model.state_dict(), os.path.join(self.save_dir, "alae_model_ckpt.pth"))
                print("Model saved.")
            
            self.train_loader.dataset.shuffle()

        return model, context_proj, losses, train_costs, test_costs

    # --------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, model, loader):
        """Compute 1D OT (emd_1d) cost per sample."""
        costs = []
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)
            Xout, Yout, M, a, b = self._pair_forward_latents(X, Y, model)

            plans = ot.emd_1d(Xout, Yout, a, b)
            cost = (plans * M).sum()
            costs.append(cost.item())
        return float(np.mean(costs))

    def _save_costs(self, train_costs, test_costs):
        """Save computed costs."""
        # pairname = f"{self.pair[0]}_{self.pair[1]}"
        pairname = "alae"
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
    from models import MLP
    from softsort import SoftSort_p2 as SoftSort
    from lapsum import soft_permutation
    from torch_geometric.datasets import ModelNet
    # from torch_geometric.loader import DataLoader
    from torch.utils.data import DataLoader
    import torch_geometric.transforms as T

    # ------------------------- CLI Arguments -------------------------
    parser = argparse.ArgumentParser(description="Amortized Min-STP Trainer")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device (cpu | cuda | cuda:IDX)")
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
    X_train, Y_train, X_test, Y_test = process_data()

    train_dataset = CustomDataset(X_train, Y_train)
    test_dataset = CustomDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)

    # ------------------------- Model Setup -------------------------

    soft_sort = SoftSort(tau=1e-2, hard=False)
    hard_sort = SoftSort(hard=True)

    # ------------------------- Trainer Init -------------------------
    trainer = AmortizedSlicerTrainer(
        soft_sort=soft_sort,
        hard_sort=hard_sort,
        soft_permutation=soft_permutation,
        device=device,
        pair=pair,
        train_loader=train_loader,
        test_loader=test_loader,
        model_class=MLP,
        lr=args.lr,
        outer_epochs=args.outer_epochs,
        inner_epochs=args.inner_epochs,
        save_dir=args.save_dir,
        seed=args.seed,
    )

    # ------------------------- Run Training -------------------------
    model, context_proj, losses, train_costs, test_costs = trainer.train()
