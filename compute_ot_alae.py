#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
amortized_slicer_trainer.py
Author: Xinran (Eva) Liu

Trainer for amortized min-STP learning on image-to-image translation.
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
from torch.utils.data import DataLoader

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
    # if not os.path.isdir('./data'):
    #     os.makedirs('./data')

    # urls = {
    #     "./data/age.npy": "https://drive.google.com/uc?id=1Vi6NzxCsS23GBNq48E-97Z9UuIuNaxPJ",
    #     "./data/gender.npy": "https://drive.google.com/uc?id=1SEdsmQGL3mOok1CPTBEfc_O1750fGRtf",
    #     "./data/latents.npy": "https://drive.google.com/uc?id=1ENhiTRsHtSjIjoRu1xYprcpNd8M9aVu8",
    #     "./data/test_images.npy": "https://drive.google.com/uc?id=1SjBWWlPjq-dxX4kxzW-Zn3iUR3po8Z0i",
    # }

    # for name, url in urls.items():
    #     gdown.download(url, os.path.join(f"{name}"), quiet=False)

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

device = "cuda:1"

X_train, Y_train, X_test, Y_test = process_data()


train_dataset = CustomDataset(X_train, Y_train)
test_dataset = CustomDataset(X_test, Y_test)
test_dataset.shuffle()

train_loader = DataLoader(train_dataset, batch_size=256)
test_loader = DataLoader(test_dataset, batch_size=256)

costs_ot= []
costs_rand= []
for X, Y in test_loader:
    X = X.to(device)
    Y = Y.to(device)

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