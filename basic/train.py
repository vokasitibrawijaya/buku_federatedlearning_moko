"""
train.py - Training dan Evaluasi untuk Federated Learning

Modul ini berisi fungsi-fungsi untuk:
1. Training lokal di klien
2. Evaluasi model
3. Helper functions untuk FL training

Author: Tim Buku Federated Learning
Date: Januari 2026
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Optional
from tqdm import tqdm
import numpy as np


def train_one_epoch(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    verbose: bool = False
) -> float:
    """
    Training untuk satu epoch.
    
    Args:
        model: Model PyTorch
        trainloader: DataLoader untuk training
        criterion: Loss function
        optimizer: Optimizer
        device: Device (CPU/GPU)
        verbose: Print progress
    
    Returns:
        Average loss untuk epoch ini
    """
    model.train()
    total_loss = 0.0
    num_batches = len(trainloader)
    
    iterator = tqdm(trainloader, desc="Training", leave=False) if verbose else trainloader
    
    for images, labels in iterator:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / num_batches


def train(
    model: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    device: torch.device,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    verbose: bool = False
) -> Dict:
    """
    Training lokal untuk satu klien (multiple epochs).
    
    Args:
        model: Model PyTorch
        trainloader: DataLoader untuk training
        epochs: Jumlah epoch
        device: Device (CPU/GPU)
        learning_rate: Learning rate
        momentum: Momentum untuk SGD
        weight_decay: Weight decay untuk regularisasi
        verbose: Print progress
    
    Returns:
        Dictionary berisi history training (loss per epoch)
    """
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )
    
    history = {'loss': []}
    
    for epoch in range(epochs):
        epoch_loss = train_one_epoch(
            model, trainloader, criterion, optimizer, device, verbose
        )
        history['loss'].append(epoch_loss)
        
        if verbose:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    return history


def train_fedprox(
    model: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    device: torch.device,
    global_params: list,
    mu: float = 0.01,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    verbose: bool = False
) -> Dict:
    """
    Training dengan FedProx (proximal term).
    
    FedProx menambahkan regularisasi untuk menjaga parameter klien
    tidak terlalu jauh dari parameter global.
    
    Args:
        model: Model PyTorch
        trainloader: DataLoader
        epochs: Jumlah epoch
        device: Device
        global_params: Parameter global dari server
        mu: Koefisien proximal term
        learning_rate: Learning rate
        momentum: Momentum
        verbose: Print progress
    
    Returns:
        Dictionary berisi history training
    """
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum
    )
    
    # Convert global params to tensors
    global_tensors = [torch.tensor(p).to(device) for p in global_params]
    
    history = {'loss': []}
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Add proximal term
            proximal_term = 0.0
            for local_param, global_param in zip(model.parameters(), global_tensors):
                proximal_term += ((local_param - global_param) ** 2).sum()
            
            loss += (mu / 2) * proximal_term
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(trainloader)
        history['loss'].append(avg_loss)
        
        if verbose:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return history


def test(
    model: nn.Module,
    testloader: DataLoader,
    device: torch.device,
    verbose: bool = False
) -> Tuple[float, float]:
    """
    Evaluasi model.
    
    Args:
        model: Model PyTorch
        testloader: DataLoader untuk testing
        device: Device (CPU/GPU)
        verbose: Print progress
    
    Returns:
        Tuple (average loss, accuracy)
    """
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        iterator = tqdm(testloader, desc="Testing", leave=False) if verbose else testloader
        
        for images, labels in iterator:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(testloader)
    accuracy = correct / total
    
    if verbose:
        print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
    
    return avg_loss, accuracy


def test_per_class(
    model: nn.Module,
    testloader: DataLoader,
    device: torch.device,
    num_classes: int = 10
) -> Dict:
    """
    Evaluasi per kelas (untuk analisis fairness).
    
    Args:
        model: Model PyTorch
        testloader: DataLoader untuk testing
        device: Device
        num_classes: Jumlah kelas
    
    Returns:
        Dictionary berisi akurasi per kelas
    """
    model.to(device)
    model.eval()
    
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    results = {
        'per_class_accuracy': [
            class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            for i in range(num_classes)
        ],
        'per_class_samples': class_total,
        'overall_accuracy': sum(class_correct) / sum(class_total),
        'worst_class_accuracy': min(
            class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            for i in range(num_classes)
        )
    }
    
    return results


def compute_gradient_norm(model: nn.Module) -> float:
    """
    Menghitung norm gradient (untuk monitoring).
    
    Args:
        model: Model PyTorch (setelah backward)
    
    Returns:
        L2 norm dari semua gradients
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


if __name__ == "__main__":
    # Test functions
    from model import SimpleCNN
    from data import load_dataset, partition_iid, get_client_dataloader
    
    print("Testing training functions...")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load small subset for testing
    trainset, testset = load_dataset('mnist')
    indices = partition_iid(trainset, num_clients=10)[0]  # Get first client's data
    trainloader = get_client_dataloader(trainset, indices[:500], batch_size=32)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)
    
    # Create model
    model = SimpleCNN(num_classes=10)
    
    # Train
    print("\nTraining for 2 epochs...")
    history = train(model, trainloader, epochs=2, device=device, verbose=True)
    print(f"Training loss: {history['loss']}")
    
    # Test
    print("\nEvaluating...")
    loss, acc = test(model, testloader, device, verbose=True)
    print(f"Test accuracy: {acc*100:.2f}%")
    
    # Per-class test
    print("\nPer-class evaluation...")
    results = test_per_class(model, testloader, device)
    print(f"Per-class accuracy: {[f'{a*100:.1f}%' for a in results['per_class_accuracy']]}")
    print(f"Worst class accuracy: {results['worst_class_accuracy']*100:.1f}%")
    
    print("\n✓ All training functions tested successfully!")
