"""
model.py - Definisi Model Neural Network untuk Federated Learning

Modul ini berisi definisi model CNN dan MLP yang digunakan dalam simulasi FL.
Model dirancang agar cukup kecil untuk simulasi di laptop, namun cukup
kompleks untuk menunjukkan konsep FL.

Author: Tim Buku Federated Learning
Date: Januari 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np


class SimpleCNN(nn.Module):
    """
    CNN sederhana untuk klasifikasi gambar MNIST (28x28, 1 channel).
    
    Arsitektur:
    - Conv1: 1 -> 32 filters, 3x3
    - Conv2: 32 -> 64 filters, 3x3
    - FC1: 64*7*7 -> 128
    - FC2: 128 -> num_classes
    
    Args:
        num_classes: Jumlah kelas output (default: 10 untuk MNIST)
    """
    
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        # FC layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class CIFAR10CNN(nn.Module):
    """
    CNN untuk klasifikasi CIFAR-10 (32x32, 3 channels).
    
    Lebih dalam dari SimpleCNN untuk menangani gambar berwarna yang
    lebih kompleks.
    
    Args:
        num_classes: Jumlah kelas output (default: 10)
    """
    
    def __init__(self, num_classes: int = 10):
        super(CIFAR10CNN, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Block 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        # Block 3
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # FC layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        
        # FC
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout2(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class SimpleMLP(nn.Module):
    """
    Multi-Layer Perceptron sederhana untuk data tabular.
    
    Digunakan untuk studi kasus seperti deteksi fraud (data tabular).
    
    Args:
        input_dim: Dimensi input
        hidden_dims: List dimensi hidden layers
        num_classes: Jumlah kelas output
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int] = [64, 32], 
        num_classes: int = 2
    ):
        super(SimpleMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def get_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function untuk membuat model.
    
    Args:
        model_name: Nama model ('cnn', 'cifar_cnn', 'mlp')
        **kwargs: Argumen tambahan untuk model
    
    Returns:
        Instance model PyTorch
    
    Example:
        >>> model = get_model('cnn', num_classes=10)
        >>> model = get_model('mlp', input_dim=30, hidden_dims=[64, 32])
    """
    models = {
        'cnn': SimpleCNN,
        'mnist_cnn': SimpleCNN,
        'cifar_cnn': CIFAR10CNN,
        'cifar10_cnn': CIFAR10CNN,
        'mlp': SimpleMLP,
    }
    
    if model_name.lower() not in models:
        raise ValueError(
            f"Model '{model_name}' tidak dikenal. "
            f"Pilihan: {list(models.keys())}"
        )
    
    return models[model_name.lower()](**kwargs)


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """
    Ekstrak parameter model sebagai list numpy arrays.
    
    Digunakan untuk mengirim parameter ke server Flower.
    
    Args:
        model: Model PyTorch
    
    Returns:
        List of numpy arrays containing model parameters
    """
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Set parameter model dari list numpy arrays.
    
    Digunakan untuk menerima parameter dari server Flower.
    
    Args:
        model: Model PyTorch
        parameters: List of numpy arrays containing new parameters
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


def count_parameters(model: nn.Module) -> int:
    """
    Hitung total parameter trainable dalam model.
    
    Args:
        model: Model PyTorch
    
    Returns:
        Jumlah parameter trainable
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    print("Testing SimpleCNN (MNIST)...")
    model = SimpleCNN(num_classes=10)
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {count_parameters(model):,}")
    
    print("\nTesting CIFAR10CNN...")
    model = CIFAR10CNN(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {count_parameters(model):,}")
    
    print("\nTesting SimpleMLP...")
    model = SimpleMLP(input_dim=30, hidden_dims=[64, 32], num_classes=2)
    x = torch.randn(2, 30)
    out = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {count_parameters(model):,}")
    
    print("\n✓ Semua model berhasil ditest!")
