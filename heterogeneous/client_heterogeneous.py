"""
client_heterogeneous.py - Client FL dengan heterogenitas sistem

Modul ini mengimplementasikan simulasi klien heterogen:
1. Resource heterogeneity (CPU/memory berbeda)
2. Network heterogeneity (bandwidth/latency berbeda)
3. Reliability heterogeneity (dropout)

Author: Tim Buku Federated Learning
Date: Januari 2026
"""

import flwr as fl
import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import random
from typing import Dict, Optional, Tuple
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basic.model import SimpleCNN, CIFAR10CNN, get_parameters, set_parameters
from basic.data import (
    load_dataset, 
    partition_dirichlet, 
    partition_iid,
    get_client_dataloader
)
from basic.train import train, train_fedprox, test


class HeterogeneousClient(fl.client.NumPyClient):
    """
    Flower Client dengan simulasi heterogenitas.
    
    Client ini mensimulasikan:
    - Slow clients (batasi compute time)
    - Network latency (tambah delay)
    - Dropout (gagal secara acak)
    
    Attributes:
        client_id: ID klien
        client_type: 'fast', 'slow', atau 'unreliable'
        compute_delay: Delay tambahan untuk simulasi slow compute
        network_delay: Delay untuk simulasi network latency
        dropout_prob: Probabilitas dropout
    """
    
    def __init__(
        self,
        client_id: int,
        model: torch.nn.Module,
        trainloader: DataLoader,
        testloader: DataLoader,
        device: torch.device,
        client_type: str = 'fast',
        compute_delay: float = 0.0,
        network_delay: float = 0.0,
        dropout_prob: float = 0.0
    ):
        """
        Initialize heterogeneous client.
        
        Args:
            client_id: ID klien
            model: Model PyTorch
            trainloader: DataLoader untuk training
            testloader: DataLoader untuk testing
            device: Device untuk training
            client_type: Tipe klien ('fast', 'slow', 'unreliable')
            compute_delay: Delay per batch (seconds)
            network_delay: Delay komunikasi (seconds)
            dropout_prob: Probabilitas dropout [0, 1]
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.client_type = client_type
        self.compute_delay = compute_delay
        self.network_delay = network_delay
        self.dropout_prob = dropout_prob
        
        print(f"[Client {client_id}] Initialized as '{client_type}' client")
        print(f"  - Compute delay: {compute_delay}s per batch")
        print(f"  - Network delay: {network_delay}s")
        print(f"  - Dropout probability: {dropout_prob*100:.1f}%")
        print(f"  - Training samples: {len(trainloader.dataset)}")
    
    def get_parameters(self, config):
        """Get current model parameters."""
        # Simulate network delay when sending parameters
        if self.network_delay > 0:
            time.sleep(self.network_delay)
        return get_parameters(self.model)
    
    def fit(self, parameters, config):
        """Training lokal dengan simulasi heterogenitas."""
        # Check for dropout
        if random.random() < self.dropout_prob:
            print(f"[Client {self.client_id}] DROPPED OUT!")
            raise Exception(f"Client {self.client_id} dropped out")
        
        # Simulate network delay receiving parameters
        if self.network_delay > 0:
            time.sleep(self.network_delay)
        
        # Set parameters
        set_parameters(self.model, parameters)
        
        # Get config
        local_epochs = config.get("local_epochs", 1)
        learning_rate = config.get("learning_rate", 0.01)
        
        print(f"[Client {self.client_id}] Training for {local_epochs} epoch(s)...")
        
        # Training with compute delay simulation
        start_time = time.time()
        
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=0.9
        )
        
        total_loss = 0.0
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Simulate compute delay
                if self.compute_delay > 0:
                    time.sleep(self.compute_delay)
            
            total_loss = epoch_loss / len(self.trainloader)
        
        training_time = time.time() - start_time
        print(f"[Client {self.client_id}] Training completed in {training_time:.2f}s")
        
        # Return updated parameters
        num_examples = len(self.trainloader.dataset)
        metrics = {
            "loss": total_loss,
            "client_id": self.client_id,
            "client_type": self.client_type,
            "training_time": training_time
        }
        
        return get_parameters(self.model), num_examples, metrics
    
    def evaluate(self, parameters, config):
        """Evaluasi model."""
        # Simulate network delay
        if self.network_delay > 0:
            time.sleep(self.network_delay)
        
        set_parameters(self.model, parameters)
        loss, accuracy = test(self.model, self.testloader, self.device)
        
        return (
            float(loss),
            len(self.testloader.dataset),
            {"accuracy": accuracy, "client_id": self.client_id}
        )


def create_heterogeneous_clients(
    num_clients: int,
    dataset_name: str = 'mnist',
    partition: str = 'dirichlet',
    alpha: float = 0.5,
    batch_size: int = 32,
    data_path: str = './data',
    seed: int = 42,
    # Heterogeneity config
    num_slow_clients: int = 3,
    num_unreliable_clients: int = 2,
    slow_delay: float = 0.01,
    network_delay: float = 0.1,
    dropout_prob: float = 0.2
) -> list:
    """
    Membuat list klien heterogen.
    
    Args:
        num_clients: Total jumlah klien
        dataset_name: 'mnist' atau 'cifar10'
        partition: 'iid' atau 'dirichlet'
        alpha: Parameter Dirichlet
        batch_size: Ukuran batch
        data_path: Path untuk data
        seed: Random seed
        num_slow_clients: Jumlah klien lambat
        num_unreliable_clients: Jumlah klien tidak stabil
        slow_delay: Delay per batch untuk klien lambat (seconds)
        network_delay: Network delay (seconds)
        dropout_prob: Probabilitas dropout untuk klien tidak stabil
    
    Returns:
        List of HeterogeneousClient instances
    """
    random.seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    trainset, testset = load_dataset(dataset_name, data_path)
    
    # Partition data
    if partition.lower() == 'iid':
        client_indices = partition_iid(trainset, num_clients, seed)
    else:
        client_indices = partition_dirichlet(trainset, num_clients, alpha, seed)
    
    # Determine client types
    client_types = ['fast'] * num_clients
    
    # Randomly assign slow clients
    slow_indices = random.sample(range(num_clients), min(num_slow_clients, num_clients))
    for idx in slow_indices:
        client_types[idx] = 'slow'
    
    # Randomly assign unreliable clients (from remaining)
    remaining = [i for i in range(num_clients) if client_types[i] == 'fast']
    unreliable_indices = random.sample(
        remaining, 
        min(num_unreliable_clients, len(remaining))
    )
    for idx in unreliable_indices:
        client_types[idx] = 'unreliable'
    
    # Create clients
    clients = []
    for client_id in range(num_clients):
        # Create dataloader
        trainloader = get_client_dataloader(
            trainset,
            client_indices[client_id],
            batch_size
        )
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        
        # Create model
        if dataset_name.lower() == 'mnist':
            model = SimpleCNN(num_classes=10)
        else:
            model = CIFAR10CNN(num_classes=10)
        
        # Set heterogeneity parameters based on type
        client_type = client_types[client_id]
        
        if client_type == 'slow':
            compute_d = slow_delay
            network_d = network_delay
            dropout_p = 0.0
        elif client_type == 'unreliable':
            compute_d = 0.0
            network_d = 0.0
            dropout_p = dropout_prob
        else:  # fast
            compute_d = 0.0
            network_d = 0.0
            dropout_p = 0.0
        
        client = HeterogeneousClient(
            client_id=client_id,
            model=model,
            trainloader=trainloader,
            testloader=testloader,
            device=device,
            client_type=client_type,
            compute_delay=compute_d,
            network_delay=network_d,
            dropout_prob=dropout_p
        )
        clients.append(client)
    
    # Summary
    print(f"\n{'='*50}")
    print("Client Heterogeneity Summary")
    print(f"{'='*50}")
    print(f"Total clients: {num_clients}")
    print(f"Fast clients: {client_types.count('fast')}")
    print(f"Slow clients: {client_types.count('slow')}")
    print(f"Unreliable clients: {client_types.count('unreliable')}")
    print(f"{'='*50}\n")
    
    return clients


def start_heterogeneous_client(
    client_id: int,
    num_clients: int = 10,
    server_address: str = "127.0.0.1:8080",
    dataset_name: str = 'mnist',
    partition: str = 'dirichlet',
    alpha: float = 0.5,
    batch_size: int = 32,
    data_path: str = './data',
    seed: int = 42,
    client_type: str = 'fast',
    slow_delay: float = 0.01,
    network_delay: float = 0.1,
    dropout_prob: float = 0.2
):
    """
    Start single heterogeneous client.
    
    Untuk menjalankan klien secara terpisah dari terminal berbeda.
    """
    random.seed(seed + client_id)
    np.random.seed(seed + client_id)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    trainset, testset = load_dataset(dataset_name, data_path)
    
    # Partition data
    if partition.lower() == 'iid':
        client_indices = partition_iid(trainset, num_clients, seed)
    else:
        client_indices = partition_dirichlet(trainset, num_clients, alpha, seed)
    
    # Create dataloader
    trainloader = get_client_dataloader(
        trainset,
        client_indices[client_id],
        batch_size
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    # Create model
    if dataset_name.lower() == 'mnist':
        model = SimpleCNN(num_classes=10)
    else:
        model = CIFAR10CNN(num_classes=10)
    
    # Set heterogeneity parameters
    if client_type == 'slow':
        compute_d = slow_delay
        network_d = network_delay
        dropout_p = 0.0
    elif client_type == 'unreliable':
        compute_d = 0.0
        network_d = 0.0
        dropout_p = dropout_prob
    else:
        compute_d = 0.0
        network_d = 0.0
        dropout_p = 0.0
    
    # Create client
    client = HeterogeneousClient(
        client_id=client_id,
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        device=device,
        client_type=client_type,
        compute_delay=compute_d,
        network_delay=network_d,
        dropout_prob=dropout_p
    )
    
    # Start client
    print(f"[Client {client_id}] Connecting to {server_address}...")
    fl.client.start_client(
        server_address=server_address,
        client=client.to_client()
    )


def main():
    parser = argparse.ArgumentParser(description='Heterogeneous Flower Client')
    
    parser.add_argument('--client-id', type=int, required=True)
    parser.add_argument('--num-clients', type=int, default=10)
    parser.add_argument('--server', type=str, default='127.0.0.1:8080')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--partition', type=str, default='dirichlet')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--client-type', type=str, default='fast',
                        choices=['fast', 'slow', 'unreliable'])
    parser.add_argument('--slow-delay', type=float, default=0.01)
    parser.add_argument('--network-delay', type=float, default=0.1)
    parser.add_argument('--dropout-prob', type=float, default=0.2)
    
    args = parser.parse_args()
    
    start_heterogeneous_client(
        client_id=args.client_id,
        num_clients=args.num_clients,
        server_address=args.server,
        dataset_name=args.dataset,
        partition=args.partition,
        alpha=args.alpha,
        batch_size=args.batch_size,
        seed=args.seed,
        client_type=args.client_type,
        slow_delay=args.slow_delay,
        network_delay=args.network_delay,
        dropout_prob=args.dropout_prob
    )


if __name__ == "__main__":
    main()
