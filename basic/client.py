"""
client.py - Flower Client untuk Federated Learning

Modul ini mengimplementasikan Flower client yang melakukan:
1. Menerima parameter global dari server
2. Training lokal pada data klien
3. Mengirim parameter update ke server

Author: Tim Buku Federated Learning
Date: Januari 2026
"""

import flwr as fl
import torch
from torch.utils.data import DataLoader
import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SimpleCNN, CIFAR10CNN, get_parameters, set_parameters
from data import load_dataset, partition_dirichlet, partition_iid, get_client_dataloader
from train import train, train_fedprox, test


class FlowerClient(fl.client.NumPyClient):
    """
    Flower Client untuk Federated Learning.
    
    Client ini melakukan training lokal dan mengirim update ke server.
    
    Attributes:
        client_id: ID unik klien
        model: Model PyTorch lokal
        trainloader: DataLoader untuk training
        testloader: DataLoader untuk testing
        device: CPU atau GPU
        config: Konfigurasi training
    """
    
    def __init__(
        self,
        client_id: int,
        model: torch.nn.Module,
        trainloader: DataLoader,
        testloader: DataLoader,
        device: torch.device,
        config: dict = None
    ):
        """
        Initialize Flower client.
        
        Args:
            client_id: ID klien
            model: Model PyTorch
            trainloader: DataLoader training
            testloader: DataLoader testing
            device: Device untuk training
            config: Konfigurasi tambahan
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.config = config or {}
        
        print(f"[Client {client_id}] Initialized with {len(trainloader.dataset)} samples")
    
    def get_parameters(self, config):
        """
        Mendapatkan parameter model saat ini.
        
        Dipanggil oleh server untuk mendapatkan parameter awal
        atau setelah training.
        
        Returns:
            List of numpy arrays berisi parameter model
        """
        return get_parameters(self.model)
    
    def fit(self, parameters, config):
        """
        Training lokal dengan parameter dari server.
        
        Args:
            parameters: Parameter global dari server
            config: Konfigurasi dari server (epochs, lr, dll)
        
        Returns:
            Tuple (parameters, num_examples, metrics)
        """
        # Set parameter dari server
        set_parameters(self.model, parameters)
        
        # Get training config
        local_epochs = config.get("local_epochs", 1)
        learning_rate = config.get("learning_rate", 0.01)
        use_fedprox = config.get("use_fedprox", False)
        mu = config.get("mu", 0.01)
        
        print(f"[Client {self.client_id}] Training for {local_epochs} epoch(s)...")
        
        # Train
        if use_fedprox:
            history = train_fedprox(
                self.model,
                self.trainloader,
                epochs=local_epochs,
                device=self.device,
                global_params=parameters,
                mu=mu,
                learning_rate=learning_rate
            )
        else:
            history = train(
                self.model,
                self.trainloader,
                epochs=local_epochs,
                device=self.device,
                learning_rate=learning_rate
            )
        
        # Return updated parameters
        num_examples = len(self.trainloader.dataset)
        metrics = {
            "loss": history['loss'][-1],
            "client_id": self.client_id
        }
        
        return get_parameters(self.model), num_examples, metrics
    
    def evaluate(self, parameters, config):
        """
        Evaluasi model dengan parameter dari server.
        
        Args:
            parameters: Parameter global dari server
            config: Konfigurasi evaluasi
        
        Returns:
            Tuple (loss, num_examples, metrics)
        """
        set_parameters(self.model, parameters)
        
        loss, accuracy = test(self.model, self.testloader, self.device)
        
        num_examples = len(self.testloader.dataset)
        metrics = {
            "accuracy": accuracy,
            "client_id": self.client_id
        }
        
        print(f"[Client {self.client_id}] Eval - Loss: {loss:.4f}, Accuracy: {accuracy*100:.2f}%")
        
        return float(loss), num_examples, metrics


def create_client(
    client_id: int,
    num_clients: int,
    dataset_name: str = 'mnist',
    partition: str = 'iid',
    alpha: float = 0.5,
    batch_size: int = 32,
    data_path: str = './data',
    seed: int = 42
) -> FlowerClient:
    """
    Factory function untuk membuat Flower client.
    
    Args:
        client_id: ID klien (0 sampai num_clients-1)
        num_clients: Total jumlah klien
        dataset_name: 'mnist' atau 'cifar10'
        partition: 'iid' atau 'dirichlet'
        alpha: Parameter Dirichlet untuk non-IID
        batch_size: Ukuran batch
        data_path: Path untuk data
        seed: Random seed
    
    Returns:
        Instance FlowerClient
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    trainset, testset = load_dataset(dataset_name, data_path)
    
    # Partition data
    if partition.lower() == 'iid':
        client_indices = partition_iid(trainset, num_clients, seed)
    else:
        client_indices = partition_dirichlet(trainset, num_clients, alpha, seed)
    
    # Create dataloaders
    trainloader = get_client_dataloader(
        trainset, 
        client_indices[client_id], 
        batch_size
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    # Create model
    if dataset_name.lower() == 'mnist':
        model = SimpleCNN(num_classes=10)
    elif dataset_name.lower() in ['cifar10', 'cifar']:
        model = CIFAR10CNN(num_classes=10)
    else:
        raise ValueError(f"Dataset '{dataset_name}' tidak didukung")
    
    # Create client
    return FlowerClient(
        client_id=client_id,
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        device=device
    )


def start_client(
    client_id: int,
    num_clients: int = 10,
    server_address: str = "127.0.0.1:8080",
    dataset_name: str = 'mnist',
    partition: str = 'iid',
    alpha: float = 0.5,
    batch_size: int = 32,
    data_path: str = './data',
    seed: int = 42
):
    """
    Start Flower client dan connect ke server.
    
    Args:
        client_id: ID klien
        num_clients: Total klien
        server_address: Alamat server (host:port)
        dataset_name: Dataset yang digunakan
        partition: Tipe partisi
        alpha: Parameter Dirichlet
        batch_size: Ukuran batch
        data_path: Path data
        seed: Random seed
    """
    # Create client
    client = create_client(
        client_id=client_id,
        num_clients=num_clients,
        dataset_name=dataset_name,
        partition=partition,
        alpha=alpha,
        batch_size=batch_size,
        data_path=data_path,
        seed=seed
    )
    
    # Start client
    print(f"[Client {client_id}] Connecting to server at {server_address}...")
    fl.client.start_client(
        server_address=server_address,
        client=client.to_client()
    )


def main():
    """Main function untuk menjalankan client dari command line."""
    parser = argparse.ArgumentParser(description='Flower FL Client')
    
    parser.add_argument(
        '--client-id', type=int, required=True,
        help='ID klien (0 sampai num_clients-1)'
    )
    parser.add_argument(
        '--num-clients', type=int, default=10,
        help='Total jumlah klien (default: 10)'
    )
    parser.add_argument(
        '--server', type=str, default='127.0.0.1:8080',
        help='Alamat server (default: 127.0.0.1:8080)'
    )
    parser.add_argument(
        '--dataset', type=str, default='mnist',
        choices=['mnist', 'cifar10'],
        help='Dataset (default: mnist)'
    )
    parser.add_argument(
        '--partition', type=str, default='iid',
        choices=['iid', 'dirichlet'],
        help='Tipe partisi data (default: iid)'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.5,
        help='Parameter Dirichlet untuk non-IID (default: 0.5)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Ukuran batch (default: 32)'
    )
    parser.add_argument(
        '--data-path', type=str, default='./data',
        help='Path untuk menyimpan data (default: ./data)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    start_client(
        client_id=args.client_id,
        num_clients=args.num_clients,
        server_address=args.server,
        dataset_name=args.dataset,
        partition=args.partition,
        alpha=args.alpha,
        batch_size=args.batch_size,
        data_path=args.data_path,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
