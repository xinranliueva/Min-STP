#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Defines dataset utilities for paired ModelNet10 classes.
"""

import os
import torch
from torch.utils.data import Dataset


def get_class_subset(dataset, class_name, raw_root="./ModelNet10/raw"):
    """
    Return a subset of ModelNet dataset belonging to a specific class.
    """
    class_names = sorted([f.name for f in os.scandir(raw_root) if f.is_dir()])
    class_idx = class_names.index(class_name)
    indices = [i for i, data in enumerate(dataset) if data.y.item() == class_idx]
    return [dataset[i] for i in indices]


class PairedModelNet(Dataset):
    """
    Construct paired datasets (class1, class2) for amortized OT alignment.

    Args:
        dataset: ModelNet dataset instance.
        class1, class2: names of two classes (e.g. 'chair', 'table').
        random_pair: if True, randomly pair samples across the two classes.
    """

    def __init__(self, dataset, class1, class2, random_pair=True):
        super().__init__()
        self.subset1 = get_class_subset(dataset, class1)
        self.subset2 = get_class_subset(dataset, class2)
        self.random_pair = random_pair
        self.len = min(len(self.subset1), len(self.subset2))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data1 = self.subset1[idx % len(self.subset1)]
        if self.random_pair:
            j = torch.randint(0, len(self.subset2), (1,)).item()
        else:
            j = idx % len(self.subset2)
        data2 = self.subset2[j]
        return data1, data2
