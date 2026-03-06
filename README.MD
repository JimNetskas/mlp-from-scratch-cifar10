# 🧠 Multi-Layer Perceptron (MLP) from Scratch | CIFAR-10

**Author:** Dimitrios Netskas

## 📌 Project Overview
This repository contains a complete, from-scratch implementation of a Multi-Layer Perceptron (MLP) built entirely using Python and **NumPy**. The objective of this project is to classify images from the **CIFAR-10 dataset** into 10 distinct categories without relying on high-level deep learning frameworks (like PyTorch or TensorFlow) for the core neural network architecture.

### 🚀 Key Results
* **Test Set Accuracy:** 57.22%
* **Best Model Weights:** Sourced from Experiment 9, Run 11 (saved as `best_model.npz`).

---

## 🏗️ Model Architecture & Mathematics
The neural network is built with the following specifications:
* **Input Layer:** 3072 neurons (flattened 3x32x32 RGB images).
* **Hidden Layers:** 2 layers utilizing a custom `Leaky_ReLU` activation function to prevent dead neurons.
* **Output Layer:** 10 neurons with a custom `softmax` activation for probability distribution.
* **Loss Function:** Custom `cross_entropy_loss` implemented with One-Hot Encoding.

---

## 📂 Repository Structure

* 📄 **`MLP.pdf`**: The comprehensive project report detailing the theoretical background, experimental setup, hyperparameter tuning, and final conclusions.
* 📓 **`mlp.ipynb`**: The main Jupyter Notebook containing the full pipeline (Data Loading, Model Definition, Training Loop, and Visualizations). *(Warning: Running this will initiate the full, time-consuming training process).*
* ⚙️ **`evaluate_model.py`**: A lightweight Python script for fast inference. It loads pre-trained weights and calculates test metrics while generating visual prediction grids.
* 💾 **`best_model.npz`**: The saved NumPy weight arrays (`W1, b1, W2, b2, W3, b3`) of the best-performing model.
* 📦 **`requirements.txt`**: Project dependencies.

---

## 💻 Installation & Quick Start (Windows)

To evaluate the model locally without retraining, follow these steps:

**1. Clone the repository & create a Virtual Environment (Recommended)**
git clone https://github.com/JimNetskas/mlp-from-scratch-cifar10.git
cd mlp-from-scratch-cifar10
python -m venv venv
venv\Scripts\activate  # Use .\venv\Scripts\Activate in PowerShell
