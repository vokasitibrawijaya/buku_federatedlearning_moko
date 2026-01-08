"""
data.py - Data Loading dan Partisi untuk Federated Learning

Modul ini menyediakan fungsi untuk:
1. Loading dataset (MNIST, CIFAR-10)
2. Partisi data ke multiple klien (IID dan Non-IID)
3. Visualisasi distribusi data

Author: Tim Buku Federated Learning
Date: Januari 2026
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os


def get_transforms(dataset_name: str) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Mendapatkan transformasi yang sesuai untuk dataset.
    
    Args:
        dataset_name: Nama dataset ('mnist' atau 'cifar10')
    
    Returns:
        Tuple (train_transform, test_transform)
    """
    if dataset_name.lower() == 'mnist':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_transform = train_transform
        
    elif dataset_name.lower() in ['cifar10', 'cifar']:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), 
                (0.2470, 0.2435, 0.2616)
            )
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), 
                (0.2470, 0.2435, 0.2616)
            )
        ])
    else:
        raise ValueError(f"Dataset '{dataset_name}' tidak didukung")
    
    return train_transform, test_transform


def load_dataset(
    dataset_name: str, 
    data_path: str = './data'
) -> Tuple[Dataset, Dataset]:
    """
    Load dataset dari torchvision.
    
    Args:
        dataset_name: Nama dataset ('mnist' atau 'cifar10')
        data_path: Path untuk menyimpan/load data
    
    Returns:
        Tuple (trainset, testset)
    """
    train_transform, test_transform = get_transforms(dataset_name)
    
    os.makedirs(data_path, exist_ok=True)
    
    if dataset_name.lower() == 'mnist':
        trainset = datasets.MNIST(
            data_path, train=True, download=True, transform=train_transform
        )
        testset = datasets.MNIST(
            data_path, train=False, download=True, transform=test_transform
        )
    elif dataset_name.lower() in ['cifar10', 'cifar']:
        trainset = datasets.CIFAR10(
            data_path, train=True, download=True, transform=train_transform
        )
        testset = datasets.CIFAR10(
            data_path, train=False, download=True, transform=test_transform
        )
    else:
        raise ValueError(f"Dataset '{dataset_name}' tidak didukung")
    
    print(f"Loaded {dataset_name.upper()}:")
    print(f"  Training samples: {len(trainset)}")
    print(f"  Test samples: {len(testset)}")
    
    return trainset, testset


def partition_iid(
    dataset: Dataset, 
    num_clients: int, 
    seed: int = 42
) -> List[List[int]]:
    """
    Partisi data secara IID (Independent and Identically Distributed).
    
    Setiap klien mendapat jumlah sample yang sama dengan distribusi
    label yang seragam.
    
    Args:
        dataset: PyTorch dataset
        num_clients: Jumlah klien
        seed: Random seed
    
    Returns:
        List of indices untuk setiap klien
    """
    np.random.seed(seed)
    
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    
    # Split equally
    split_indices = np.array_split(indices, num_clients)
    client_indices = [list(idx) for idx in split_indices]
    
    return client_indices


def partition_dirichlet(
    dataset: Dataset, 
    num_clients: int, 
    alpha: float = 0.5,
    seed: int = 42
) -> List[List[int]]:
    """
    Partisi data menggunakan distribusi Dirichlet (Non-IID).
    
    Alpha mengontrol tingkat heterogenitas:
    - alpha -> 0: Sangat non-IID (setiap klien hanya punya 1-2 kelas)
    - alpha -> inf: Mendekati IID
    - alpha = 1: Uniform distribution
    
    Args:
        dataset: PyTorch dataset
        num_clients: Jumlah klien
        alpha: Parameter Dirichlet (semakin kecil = semakin non-IID)
        seed: Random seed
    
    Returns:
        List of indices untuk setiap klien
    """
    np.random.seed(seed)
    
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        raise ValueError("Dataset tidak memiliki atribut 'targets' atau 'labels'")
    
    num_classes = len(np.unique(labels))
    
    # Get indices per class
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    # Initialize client indices
    client_indices = [[] for _ in range(num_clients)]
    
    # Distribute samples per class using Dirichlet
    for class_idx in range(num_classes):
        indices = class_indices[class_idx].copy()
        np.random.shuffle(indices)
        
        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet([alpha] * num_clients)
        
        # Calculate number of samples per client
        num_samples_per_client = (proportions * len(indices)).astype(int)
        
        # Adjust for rounding errors
        diff = len(indices) - num_samples_per_client.sum()
        num_samples_per_client[np.argmax(proportions)] += diff
        
        # Distribute
        start = 0
        for client_id in range(num_clients):
            num_samples = num_samples_per_client[client_id]
            client_indices[client_id].extend(
                indices[start:start + num_samples].tolist()
            )
            start += num_samples
    
    return client_indices


def partition_quantity_skew(
    dataset: Dataset,
    num_clients: int,
    min_samples: int = 100,
    seed: int = 42
) -> List[List[int]]:
    """
    Partisi dengan quantity skew (jumlah sample berbeda per klien).
    
    Args:
        dataset: PyTorch dataset
        num_clients: Jumlah klien
        min_samples: Minimum samples per klien
        seed: Random seed
    
    Returns:
        List of indices untuk setiap klien
    """
    np.random.seed(seed)
    
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    
    # Generate random proportions
    proportions = np.random.dirichlet([1] * num_clients)
    
    # Ensure minimum samples
    num_per_client = (proportions * num_samples).astype(int)
    num_per_client = np.maximum(num_per_client, min_samples)
    
    # Adjust if total exceeds available
    while num_per_client.sum() > num_samples:
        idx = np.argmax(num_per_client)
        num_per_client[idx] -= 1
    
    # Distribute
    client_indices = []
    start = 0
    for i in range(num_clients):
        end = start + num_per_client[i]
        client_indices.append(indices[start:end].tolist())
        start = end
    
    return client_indices


def get_partition_stats(
    dataset: Dataset, 
    client_indices: List[List[int]]
) -> Dict:
    """
    Menghitung statistik distribusi data per klien.
    
    Args:
        dataset: PyTorch dataset
        client_indices: List of indices per klien
    
    Returns:
        Dictionary berisi statistik
    """
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array(dataset.labels)
    
    num_classes = len(np.unique(labels))
    num_clients = len(client_indices)
    
    stats = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'samples_per_client': [],
        'label_distribution': [],
    }
    
    for client_id, indices in enumerate(client_indices):
        client_labels = labels[indices]
        
        # Count per class
        class_counts = np.zeros(num_classes, dtype=int)
        for label in client_labels:
            class_counts[label] += 1
        
        stats['samples_per_client'].append(len(indices))
        stats['label_distribution'].append(class_counts.tolist())
    
    return stats


def visualize_partition(
    dataset: Dataset,
    client_indices: List[List[int]],
    save_path: Optional[str] = None
) -> None:
    """
    Visualisasi distribusi data per klien.
    
    Args:
        dataset: PyTorch dataset
        client_indices: List of indices per klien
        save_path: Path untuk menyimpan gambar (opsional)
    """
    stats = get_partition_stats(dataset, client_indices)
    
    num_clients = stats['num_clients']
    num_classes = stats['num_classes']
    distribution = np.array(stats['label_distribution'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap distribusi label
    im = axes[0].imshow(distribution.T, aspect='auto', cmap='Blues')
    axes[0].set_xlabel('Client ID')
    axes[0].set_ylabel('Class')
    axes[0].set_title('Label Distribution per Client')
    axes[0].set_xticks(range(num_clients))
    axes[0].set_yticks(range(num_classes))
    plt.colorbar(im, ax=axes[0], label='Number of Samples')
    
    # Bar chart jumlah sample per klien
    axes[1].bar(range(num_clients), stats['samples_per_client'], color='steelblue')
    axes[1].set_xlabel('Client ID')
    axes[1].set_ylabel('Number of Samples')
    axes[1].set_title('Samples per Client')
    axes[1].set_xticks(range(num_clients))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def get_client_dataloader(
    dataset: Dataset,
    indices: List[int],
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    """
    Membuat DataLoader untuk satu klien.
    
    Args:
        dataset: PyTorch dataset
        indices: Indices milik klien ini
        batch_size: Ukuran batch
        shuffle: Apakah shuffle data
    
    Returns:
        DataLoader untuk klien
    """
    subset = Subset(dataset, indices)
    return DataLoader(
        subset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0,  # Set 0 untuk kompatibilitas Windows
        pin_memory=True
    )


def prepare_data_for_clients(
    dataset_name: str,
    num_clients: int,
    partition: str = 'iid',
    alpha: float = 0.5,
    batch_size: int = 32,
    data_path: str = './data',
    seed: int = 42
) -> Tuple[List[DataLoader], DataLoader, List[List[int]]]:
    """
    One-stop function untuk menyiapkan data untuk semua klien.
    
    Args:
        dataset_name: 'mnist' atau 'cifar10'
        num_clients: Jumlah klien
        partition: 'iid', 'dirichlet', atau 'quantity'
        alpha: Parameter untuk Dirichlet (jika partition='dirichlet')
        batch_size: Ukuran batch
        data_path: Path data
        seed: Random seed
    
    Returns:
        Tuple (list of train loaders, test loader, client_indices)
    """
    # Load dataset
    trainset, testset = load_dataset(dataset_name, data_path)
    
    # Partition
    if partition.lower() == 'iid':
        client_indices = partition_iid(trainset, num_clients, seed)
    elif partition.lower() in ['dirichlet', 'noniid', 'non-iid']:
        client_indices = partition_dirichlet(trainset, num_clients, alpha, seed)
    elif partition.lower() in ['quantity', 'quantity_skew']:
        client_indices = partition_quantity_skew(trainset, num_clients, seed=seed)
    else:
        raise ValueError(f"Partition '{partition}' tidak didukung")
    
    # Create dataloaders
    train_loaders = [
        get_client_dataloader(trainset, indices, batch_size)
        for indices in client_indices
    ]
    
    test_loader = DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Print stats
    stats = get_partition_stats(trainset, client_indices)
    print(f"\nPartition type: {partition}")
    print(f"Samples per client: min={min(stats['samples_per_client'])}, "
          f"max={max(stats['samples_per_client'])}, "
          f"mean={np.mean(stats['samples_per_client']):.1f}")
    
    return train_loaders, test_loader, client_indices


if __name__ == "__main__":
    # Demo partisi
    print("=" * 60)
    print("Demo Data Partitioning")
    print("=" * 60)
    
    # Load MNIST
    trainset, testset = load_dataset('mnist')
    
    # Test IID partition
    print("\n--- IID Partition ---")
    iid_indices = partition_iid(trainset, num_clients=10)
    stats = get_partition_stats(trainset, iid_indices)
    print(f"Samples per client: {stats['samples_per_client']}")
    
    # Test Dirichlet partition (non-IID)
    print("\n--- Dirichlet Partition (alpha=0.5) ---")
    noniid_indices = partition_dirichlet(trainset, num_clients=10, alpha=0.5)
    stats = get_partition_stats(trainset, noniid_indices)
    print(f"Samples per client: {stats['samples_per_client']}")
    
    # Visualize
    print("\n--- Visualizing Non-IID Partition ---")
    visualize_partition(trainset, noniid_indices)
    
    print("\n✓ Demo selesai!")
