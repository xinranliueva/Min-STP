# 🌀 Amortized Min-STP Training for 3D Point Cloud Matching

This repository implements the **Amortized Minimum Sliced Transport Plan
(Min-STP)** framework for learning efficient alignment between 3D point
clouds using **PointNet-based autoencoding** and **soft permutation
layers**.

The implementation is modular and reproducible, designed for experiments
on the **ModelNet10** dataset.

------------------------------------------------------------------------

## 📁 Directory Structure

    /home/liux31/Min_Gen_Slices/pointcloud/
    │
    ├── amortized_slicer.py        # Main training script (command-line executable)
    ├── pointcloud_ae.py           # PointNet-style AutoEncoder for 3D point clouds
    ├── data.py                    # Dataset utilities: PairedModelNet, get_class_subset
    ├── models.py                  # Model options used for amortized STP mapping
    ├── softsort.py                # Differentiable SoftSort operator implementation
    └── lapsum.py                  # Laplace-summed soft permutation operator (soft_permutation)

------------------------------------------------------------------------

## ⚙️ Requirements

Python ≥ 3.10 with the following key packages:

``` bash
torch
torch-geometric
numpy
tqdm
matplotlib
seaborn
pot          # Python Optimal Transport (ot)
```

------------------------------------------------------------------------

## 🧠 Overview

The amortized Min-STP framework learns a **contextualized transport
map** between paired 3D point clouds: - A **PointNet AutoEncoder**
encodes each shape into a latent context vector. - A lightweight **MLP
with skip connections** learns slice-wise transport scores. - The model
is trained with **two symmetric soft/hard sorting passes**, forming a
differentiable approximation of the **minimum sliced transport plan**.

------------------------------------------------------------------------

## 🧩 Core Components

  -----------------------------------------------------------------------------
  File                        Description
  --------------------------- -------------------------------------------------
  **`pointcloud_ae.py`**      Defines `PointCloudAE`, a PointNet-style
                              encoder--decoder for point clouds.

  **`data.py`**               Defines `PairedModelNet`, which constructs class
                              pairs (e.g., *chair* vs *table*) for OT
                              alignment.

  **`models.py`**             Defines `MLPWithSkipConnections`, the amortized
                              map network used to predict slice outputs.

  **`softsort.py`**           Provides differentiable sorting operators
                              (`SoftSort_p2` and hard variants).

  **`lapsum.py`**             Implements `soft_permutation`, a Laplace-smoothed
                              soft matching operator.

  **`amortized_slicer.py`**   Main entry point: orchestrates training,
                              evaluation, and result saving.
  -----------------------------------------------------------------------------

------------------------------------------------------------------------

## 📦 Dataset: ModelNet10

The script automatically downloads and preprocesses **ModelNet10** via
PyTorch Geometric:

-   Each object is sampled into **1024 points**
    (`T.SamplePoints(1024)`).
-   Coordinates are normalized via `T.NormalizeScale()`.
-   You can choose any two classes (e.g., `chair` vs `table`, `monitor`
    vs `desk`) for training.

Raw data is typically stored under:

    ~/Min_Gen_Slices/pointcloud/ModelNet10/

------------------------------------------------------------------------

## 🚀 Usage

### 1. Train amortized Min-STP between two classes

``` bash
python amortized_slicer.py --device cuda:0 --pair chair table
```

This will: - Load pre-trained AE from
`./ModelNet10/model_checkpoint.pth` - Train amortized min-STP for 25
outer epochs × 100 inner iterations - Save results under `./ckpts/`

### 2. Optional arguments

  Argument           Default          Description
  ------------------ ---------------- --------------------------------------------
  `--device`         `cuda:0`         Compute device (`cpu`, `cuda`, `cuda:IDX`)
  `--pair`           `monitor desk`   Pair of ModelNet10 classes
  `--lr`             `1e-4`           Learning rate
  `--outer_epochs`   `25`             Number of outer epochs
  `--inner_epochs`   `100`            Inner updates per batch
  `--save_dir`       `./ckpts`        Directory to save results
  `--seed`           `42`             Random seed for reproducibility

Example with custom setup:

``` bash
python amortized_slicer.py --device cuda:1 --pair chair toilet --outer_epochs 30 --inner_epochs 120 --lr 5e-5
```

------------------------------------------------------------------------

## 🧾 Outputs

After training, results are stored in `ckpts/`:

    ckpts/
    ├── train_costs_chair_table.pkl
    ├── test_costs_chair_table.pkl

Each `.pkl` file contains a NumPy array of averaged costs per evaluation
epoch.\
You can visualize correlations with OT or other baselines (see your
original plotting scripts).

------------------------------------------------------------------------

## 🔬 Extensions

This modular structure supports: - Plug-in replacements for the encoder
(e.g., DGCNN, PointTransformer) - Alternate differentiable sorting
(e.g., SinkhornSort) - New contexts (e.g., temporal slices, multimodal
features)

------------------------------------------------------------------------

## 🧑‍💻 Example Workflow

1.  **Pretrain AutoEncoder**

    ``` bash
    python pretrain_pointcloud_ae.py
    ```

    (if you wish to retrain the AE instead of loading
    `model_checkpoint.pth`)

2.  **Train Amortized STP**

    ``` bash
    python amortized_slicer.py --pair chair table
    ```

3.  **Analyze and Plot** Load `.pkl` results for comparison with
    baseline OT or Wasserstein Wormhole distances.

------------------------------------------------------------------------

## ✨ Citation

If you use this codebase in academic work, please cite:

    @inproceedings{liu2025amortizedstp,
      title={Amortized Minimum Sliced Transport Plan for 3D Shape Alignment},
      author={Liu, Xinran and Kolouri, Soheil},
      booktitle={ICLR},
      year={2025}
    }

------------------------------------------------------------------------

## 📧 Contact

Maintained by **Xinran (Eva) Liu**\
Ph.D. Candidate, Computer Science @ Vanderbilt University\
✉️ \[email address / GitHub profile if desired\]
