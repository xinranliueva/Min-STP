# Efficient Transferable Optimal Transport via Min-Sliced Transport Plans 

This repository provides modular components for learning Min-STP. It includes:

- **Amortized training pipelines** (e.g., Gaussian → shape, class → class)  
- **Sequential / multi-task training pipelines** (e.g., gradual task drift via `SequenceTransfer.ipynb`)  
- PointNet autoencoders (for context vectors)
- SetTransformer / MLP models  
- Differentiable sorting and soft permutation operators  
- Data utilities for ModelNet10

The structure is designed so each module can be reused independently.

---

## 📁 Directory Overview

```
./
├── README.md
│
├── amortized_slicer.py
├── amortized_slicer_pc.py
│
├── pointcloud_ae.py
├── data.py
├── models.py
├── softsort.py
├── lapsum.py
│
├── SequenceTransfer.ipynb
└── ModelNet10/
```

---

## 🔧 File Descriptions

### **Training Pipelines**

- **`amortized_slicer.py`**  
  Paired **class → class** training script using symmetric soft/hard sorting.  
  Uses latent context vectors from the point cloud autoencoder model.

- **`amortized_slicer_pc.py`**  
  **Gaussian → shape** amortized training.  
  Uses Gaussian noise as the source distribution and conditions on a shape’s latent context vector.

- **`SequenceTransfer.ipynb`**  
  Sequential training across **gradually drifting tasks**.  
  Useful for transferability, curriculum learning, and studying how learned slicers adapt across sequences of related distributions.

---

### **Models**

- **`pointcloud_ae.py`**  
  PointNet-style autoencoder that produces latent embeddings for 3D point clouds.

- **`models.py`**  
  Contains all slicer and set-based architectures:  
  - `SetTransformer` (default)  
  - `MLPWithSkipConnections`, `MLP`, etc.
  - `pairSetTransformer` for cross-set attention  
  - Residual and multi-head attention blocks

---

### **Differentiable Operators**

- **`softsort.py`**  
  SoftSort operator (soft and hard variants).  
  Used only for hard sorting in the code.

- **`lapsum.py`**  
  Laplace-summed `soft_permutation` operator.  
  Produces stable soft permutation matrices.

---

### **Dataset Utilities**

- **`data.py`**  
  - `PairedModelNet`: builds paired samples for two ModelNet10 classes  
  - `get_class_subset`: extracts a class-specific subset  

---

## 🚀 Usage

### **Paired class → class**

```bash
python amortized_slicer.py --device cuda:0 --pair chair table
```

### **Gaussian → shape**

```bash
python amortized_slicer_pc.py --device cuda:0
```

Both scripts:
- Load ModelNet10 via PyTorch Geometric  
- Load the autoencoder  
- Train a slicer model  
- Save outputs in `./ckpts/`

---

## 📦 Outputs

```
ckpts/
├── pointnet_model_ckpt.pth
├── train_costs_*.pkl
└── test_costs_*.pkl
```
