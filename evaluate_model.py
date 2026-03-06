import numpy as np
import os
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

# --- 1. Ρυθμίσεις ---
MODEL_PATH = "best_model.npz"
DATA_DIR = "./data"

# --- 2. Μαθηματικές Συναρτήσεις ---
def Leaky_ReLU(Z):
    return np.maximum(0.01 * Z, Z)

def softmax(Z):
    Zs = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Zs)
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def one_hot(Y, num_classes=10):
    oh = np.zeros((num_classes, Y.size), dtype=np.float32)
    oh[Y, np.arange(Y.size)] = 1.0
    return oh

def cross_entropy_loss(A3, Y):
    m = Y.size
    oh = one_hot(Y)
    return -np.sum(np.log(A3 + 1e-12) * oh) / m

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Forward Prop (Inference only - No Dropout)
def forward_prop_inference(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = Leaky_ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = Leaky_ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return A3

# --- 3. Φόρτωση Δεδομένων ---
def load_all_data():
    print("Loading CIFAR-10 Data (Train & Test)...")
    transform = T.ToTensor() # [0,1] Normalization
    
    # Training Set
    train_set = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    X_train = np.stack([img.numpy() for (img, _) in train_set]).astype(np.float32)
    y_train = np.array([lbl for (_, lbl) in train_set], dtype=np.int64)
    X_train = X_train.reshape(X_train.shape[0], -1).T  # (3072, 50000)

    # Test Set
    test_set = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
    X_test = np.stack([img.numpy() for (img, _) in test_set]).astype(np.float32)
    y_test = np.array([lbl for (_, lbl) in test_set], dtype=np.int64)
    X_test = X_test.reshape(X_test.shape[0], -1).T    # (3072, 10000)
    
    print(f"Data loaded: {X_train.shape[1]} train images, {X_test.shape[1]} test images.")
    return X_train, y_train, X_test, y_test

# --- 4. Plotting Utils ---
def plot_grid(X, y_true, preds, indices, title, classes):
    plt.figure(figsize=(12, 5))
    plt.suptitle(title, fontsize=14, fontweight='bold')
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        img = X[:, idx].reshape(3, 32, 32).transpose(1, 2, 0)
        plt.imshow(img)
        col = 'green' if preds[idx] == y_true[idx] else 'red'
        plt.title(f"T: {classes[y_true[idx]]}\nP: {classes[preds[idx]]}", color=col, fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_bar_chart(classes, accuracies):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', edgecolor='black')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Test Accuracy')
    plt.ylim(0, 100)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# --- 5. Main Execution ---
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: File '{MODEL_PATH}' not found.")
        return

    # 1. Load Data
    X_train, Y_train, X_test, Y_test = load_all_data()

    # 2. Load Model
    print(f"Loading model weights from {MODEL_PATH}...")
    try:
        data = np.load(MODEL_PATH)
        W1, b1 = data['W1'], data['b1']
        W2, b2 = data['W2'], data['b2']
        W3, b3 = data['W3'], data['b3']
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Calculate Metrics (Forward Pass)
    print("Calculating metrics on Training Set (this may take a moment)...")
    A3_train = forward_prop_inference(W1, b1, W2, b2, W3, b3, X_train)
    preds_train = get_predictions(A3_train)
    acc_train = get_accuracy(preds_train, Y_train)
    loss_train = cross_entropy_loss(A3_train, Y_train)

    print("Calculating metrics on Test Set...")
    A3_test = forward_prop_inference(W1, b1, W2, b2, W3, b3, X_test)
    preds_test = get_predictions(A3_test)
    acc_test = get_accuracy(preds_test, Y_test)
    loss_test = cross_entropy_loss(A3_test, Y_test)

    # 4. Print Summary Table
    print("\n" + "="*45)
    print(f"{'METRIC':<15} | {'TRAIN':<12} | {'TEST':<12}")
    print("="*45)
    print(f"{'Accuracy':<15} | {acc_train*100:.2f}%{'':<6} | {acc_test*100:.2f}%")
    print(f"{'Loss':<15} | {loss_train:.4f}{'':<8} | {loss_test:.4f}")
    print("="*45 + "\n")

    # 5. Per-Class Accuracy (Test Set)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    class_accs = []
    print(f"{'Class':<10} | Accuracy")
    print("-" * 25)
    for i, class_name in enumerate(classes):
        class_indices = np.where(Y_test == i)[0]
        if len(class_indices) > 0:
            class_acc = np.mean(preds_test[class_indices] == Y_test[class_indices]) * 100
            class_accs.append(class_acc)
            print(f"{class_name:<10} | {class_acc:.2f}%")
        else:
            class_accs.append(0.0)
    print("-" * 40)

    # 6. Visualizations
    print("Displaying Plots...")
    plot_bar_chart(classes, class_accs)

    # 10 Correct
    correct_idxs = np.where(preds_test == Y_test)[0]
    if len(correct_idxs) >= 10:
        sel = np.random.choice(correct_idxs, 10, replace=False)
        plot_grid(X_test, Y_test, preds_test, sel, "10 Correct Predictions", classes)

    # 10 Incorrect
    wrong_idxs = np.where(preds_test != Y_test)[0]
    if len(wrong_idxs) >= 10:
        sel = np.random.choice(wrong_idxs, 10, replace=False)
        plot_grid(X_test, Y_test, preds_test, sel, "10 Incorrect Predictions", classes)

if __name__ == "__main__":
    main()