"""
server.py - Flower Server untuk Federated Learning

Modul ini mengimplementasikan Flower server yang:
1. Mengkoordinasi training dengan multiple klien
2. Mengagregasi parameter (FedAvg, FedProx, dll)
3. Logging dan monitoring

Author: Tim Buku Federated Learning
Date: Januari 2026
"""

import flwr as fl
from flwr.common import Metrics, Parameters, ndarrays_to_parameters
from flwr.server.strategy import FedAvg, FedProx
from typing import List, Tuple, Optional, Dict, Callable
import argparse
import json
import os
from datetime import datetime
import numpy as np
import torch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SimpleCNN, CIFAR10CNN, get_parameters


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Menghitung weighted average dari metrics klien.
    
    Digunakan untuk agregasi akurasi/loss dari multiple klien,
    dengan weight berdasarkan jumlah sampel.
    
    Args:
        metrics: List of (num_examples, metrics_dict) dari klien
    
    Returns:
        Dictionary berisi aggregated metrics
    """
    # Aggregate accuracy
    if not metrics:
        return {}
    
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    result = {}
    
    # Check if accuracy exists
    if 'accuracy' in metrics[0][1]:
        weighted_acc = sum(
            num_examples * m.get('accuracy', 0) 
            for num_examples, m in metrics
        )
        result['accuracy'] = weighted_acc / total_examples
    
    # Check if loss exists
    if 'loss' in metrics[0][1]:
        weighted_loss = sum(
            num_examples * m.get('loss', 0)
            for num_examples, m in metrics
        )
        result['loss'] = weighted_loss / total_examples
    
    return result


class LoggingStrategy(FedAvg):
    """
    Strategy dengan enhanced logging.
    
    Extends FedAvg dengan logging detail per ronde.
    
    Attributes:
        history: Dictionary menyimpan metrics per ronde
        log_dir: Directory untuk menyimpan logs
    """
    
    def __init__(
        self,
        log_dir: str = './logs',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.history = {
            'round': [],
            'loss': [],
            'accuracy': [],
            'num_clients': [],
            'num_failures': []
        }
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
    
    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures
    ):
        """Override aggregate_fit untuk logging."""
        # Log statistics
        num_success = len(results)
        num_fail = len(failures)
        
        print(f"\n{'='*50}")
        print(f"Round {server_round}: {num_success} success, {num_fail} failures")
        
        # Log individual client results
        for client_proxy, fit_res in results:
            client_id = fit_res.metrics.get('client_id', 'unknown')
            loss = fit_res.metrics.get('loss', 'N/A')
            print(f"  Client {client_id}: loss={loss:.4f}" if isinstance(loss, float) else f"  Client {client_id}: loss={loss}")
        
        # Store stats
        self.history['round'].append(server_round)
        self.history['num_clients'].append(num_success)
        self.history['num_failures'].append(num_fail)
        
        # Call parent
        return super().aggregate_fit(server_round, results, failures)
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures
    ):
        """Override aggregate_evaluate untuk logging."""
        if not results:
            return None, {}
        
        # Aggregate
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        
        # Calculate weighted accuracy
        total_examples = sum(num for _, num, _ in results)
        weighted_acc = sum(
            num * metrics.get('accuracy', 0)
            for _, num, metrics in results
        ) / total_examples
        
        weighted_loss = sum(
            num * loss
            for loss, num, _ in results
        ) / total_examples
        
        print(f"Aggregated - Loss: {weighted_loss:.4f}, Accuracy: {weighted_acc*100:.2f}%")
        print(f"{'='*50}\n")
        
        # Store
        self.history['loss'].append(weighted_loss)
        self.history['accuracy'].append(weighted_acc)
        
        return aggregated
    
    def save_history(self, filename: str = 'training_history.json'):
        """Simpan history ke file JSON."""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved to {filepath}")


def get_on_fit_config(
    local_epochs: int = 1,
    learning_rate: float = 0.01,
    use_fedprox: bool = False,
    mu: float = 0.01
) -> Callable[[int], Dict]:
    """
    Factory untuk membuat function on_fit_config.
    
    Args:
        local_epochs: Jumlah epoch lokal per ronde
        learning_rate: Learning rate
        use_fedprox: Gunakan FedProx
        mu: Koefisien proximal untuk FedProx
    
    Returns:
        Function yang mengembalikan config dict
    """
    def fit_config(server_round: int) -> Dict:
        """Return training config untuk ronde tertentu."""
        config = {
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
            "server_round": server_round,
            "use_fedprox": use_fedprox,
            "mu": mu
        }
        return config
    
    return fit_config


def get_initial_parameters(
    dataset_name: str = 'mnist'
) -> Parameters:
    """
    Mendapatkan parameter awal dari model.
    
    Args:
        dataset_name: 'mnist' atau 'cifar10'
    
    Returns:
        Flower Parameters object
    """
    if dataset_name.lower() == 'mnist':
        model = SimpleCNN(num_classes=10)
    elif dataset_name.lower() in ['cifar10', 'cifar']:
        model = CIFAR10CNN(num_classes=10)
    else:
        raise ValueError(f"Dataset '{dataset_name}' tidak didukung")
    
    return ndarrays_to_parameters(get_parameters(model))


def start_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = 10,
    fraction_fit: float = 0.5,
    fraction_evaluate: float = 0.5,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    local_epochs: int = 1,
    learning_rate: float = 0.01,
    dataset_name: str = 'mnist',
    use_fedprox: bool = False,
    mu: float = 0.01,
    log_dir: str = './logs'
):
    """
    Start Flower server.
    
    Args:
        server_address: Alamat server (host:port)
        num_rounds: Jumlah ronde training
        fraction_fit: Fraksi klien untuk training per ronde
        fraction_evaluate: Fraksi klien untuk evaluasi
        min_fit_clients: Minimum klien untuk training
        min_evaluate_clients: Minimum klien untuk evaluasi
        min_available_clients: Minimum klien yang harus tersedia
        local_epochs: Epoch lokal per ronde
        learning_rate: Learning rate
        dataset_name: Dataset yang digunakan
        use_fedprox: Gunakan FedProx
        mu: Koefisien proximal
        log_dir: Directory untuk logs
    """
    print(f"\n{'='*60}")
    print("FLOWER FEDERATED LEARNING SERVER")
    print(f"{'='*60}")
    print(f"Server address: {server_address}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Fraction fit: {fraction_fit}")
    print(f"Min clients: {min_available_clients}")
    print(f"Local epochs: {local_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Dataset: {dataset_name}")
    print(f"Strategy: {'FedProx (mu={mu})' if use_fedprox else 'FedAvg'}")
    print(f"{'='*60}\n")
    
    # Get initial parameters
    initial_parameters = get_initial_parameters(dataset_name)
    
    # Create strategy
    strategy = LoggingStrategy(
        log_dir=log_dir,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=get_on_fit_config(
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            use_fedprox=use_fedprox,
            mu=mu
        ),
        initial_parameters=initial_parameters
    )
    
    # Start server
    print("Waiting for clients to connect...")
    
    history = fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )
    
    # Save history
    strategy.save_history()
    
    return history, strategy.history


def main():
    """Main function untuk menjalankan server dari command line."""
    parser = argparse.ArgumentParser(description='Flower FL Server')
    
    parser.add_argument(
        '--address', type=str, default='0.0.0.0:8080',
        help='Alamat server (default: 0.0.0.0:8080)'
    )
    parser.add_argument(
        '--num-rounds', type=int, default=10,
        help='Jumlah ronde training (default: 10)'
    )
    parser.add_argument(
        '--fraction-fit', type=float, default=0.5,
        help='Fraksi klien per ronde (default: 0.5)'
    )
    parser.add_argument(
        '--min-clients', type=int, default=2,
        help='Minimum klien yang dibutuhkan (default: 2)'
    )
    parser.add_argument(
        '--local-epochs', type=int, default=1,
        help='Epoch lokal per ronde (default: 1)'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.01,
        help='Learning rate (default: 0.01)'
    )
    parser.add_argument(
        '--dataset', type=str, default='mnist',
        choices=['mnist', 'cifar10'],
        help='Dataset (default: mnist)'
    )
    parser.add_argument(
        '--fedprox', action='store_true',
        help='Gunakan FedProx instead of FedAvg'
    )
    parser.add_argument(
        '--mu', type=float, default=0.01,
        help='Koefisien proximal untuk FedProx (default: 0.01)'
    )
    parser.add_argument(
        '--log-dir', type=str, default='./logs',
        help='Directory untuk logs (default: ./logs)'
    )
    
    args = parser.parse_args()
    
    start_server(
        server_address=args.address,
        num_rounds=args.num_rounds,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_fit,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        dataset_name=args.dataset,
        use_fedprox=args.fedprox,
        mu=args.mu,
        log_dir=args.log_dir
    )


if __name__ == "__main__":
    main()
