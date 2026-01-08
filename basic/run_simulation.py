"""
run_simulation.py - Script untuk menjalankan simulasi FL lengkap

Script ini memungkinkan menjalankan simulasi FL dalam satu proses
tanpa perlu membuka multiple terminal. Cocok untuk:
1. Quick testing dan debugging
2. Eksperimen dengan parameter berbeda
3. Automasi batch experiments

Author: Tim Buku Federated Learning
Date: Januari 2026
"""

import flwr as fl
from flwr.common import ndarrays_to_parameters, Metrics
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation

import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

from model import SimpleCNN, CIFAR10CNN, get_parameters, set_parameters
from data import (
    load_dataset, 
    partition_iid, 
    partition_dirichlet,
    get_client_dataloader,
    visualize_partition,
    get_partition_stats
)
from train import train, test


# Global variables for simulation
trainset = None
testset = None
client_indices = None
device = None


def get_client_fn(
    dataset_name: str,
    batch_size: int,
    local_epochs: int,
    learning_rate: float
):
    """
    Factory untuk membuat client_fn yang digunakan dalam simulasi.
    
    Args:
        dataset_name: Nama dataset
        batch_size: Ukuran batch
        local_epochs: Epoch lokal
        learning_rate: Learning rate
    
    Returns:
        Function client_fn(cid) -> FlowerClient
    """
    global trainset, testset, client_indices, device
    
    class SimulationClient(fl.client.NumPyClient):
        def __init__(self, cid: str):
            self.cid = int(cid)
            
            # Create model
            if dataset_name.lower() == 'mnist':
                self.model = SimpleCNN(num_classes=10)
            else:
                self.model = CIFAR10CNN(num_classes=10)
            
            self.model.to(device)
            
            # Create dataloader
            self.trainloader = get_client_dataloader(
                trainset, 
                client_indices[self.cid],
                batch_size
            )
            self.testloader = DataLoader(
                testset, 
                batch_size=batch_size, 
                shuffle=False
            )
        
        def get_parameters(self, config):
            return get_parameters(self.model)
        
        def fit(self, parameters, config):
            set_parameters(self.model, parameters)
            
            epochs = config.get("local_epochs", local_epochs)
            lr = config.get("learning_rate", learning_rate)
            
            history = train(
                self.model, 
                self.trainloader, 
                epochs=epochs,
                device=device,
                learning_rate=lr
            )
            
            return (
                get_parameters(self.model),
                len(self.trainloader.dataset),
                {"loss": history['loss'][-1], "client_id": self.cid}
            )
        
        def evaluate(self, parameters, config):
            set_parameters(self.model, parameters)
            loss, accuracy = test(self.model, self.testloader, device)
            return (
                float(loss),
                len(self.testloader.dataset),
                {"accuracy": accuracy, "client_id": self.cid}
            )
    
    def client_fn(cid: str) -> fl.client.NumPyClient:
        return SimulationClient(cid)
    
    return client_fn


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Weighted average untuk metrics."""
    if not metrics:
        return {}
    
    total_examples = sum(n for n, _ in metrics)
    
    result = {}
    if 'accuracy' in metrics[0][1]:
        acc = sum(n * m['accuracy'] for n, m in metrics) / total_examples
        result['accuracy'] = acc
    
    return result


def run_simulation(
    num_clients: int = 10,
    num_rounds: int = 10,
    dataset_name: str = 'mnist',
    partition: str = 'iid',
    alpha: float = 0.5,
    fraction_fit: float = 0.5,
    local_epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    seed: int = 42,
    data_path: str = './data',
    output_dir: str = './results',
    visualize: bool = True
) -> Dict:
    """
    Menjalankan simulasi FL lengkap.
    
    Args:
        num_clients: Jumlah klien
        num_rounds: Jumlah ronde training
        dataset_name: 'mnist' atau 'cifar10'
        partition: 'iid' atau 'dirichlet'
        alpha: Parameter Dirichlet (untuk non-IID)
        fraction_fit: Fraksi klien per ronde
        local_epochs: Epoch lokal per ronde
        batch_size: Ukuran batch
        learning_rate: Learning rate
        seed: Random seed
        data_path: Path untuk data
        output_dir: Directory untuk output
        visualize: Plot hasil di akhir
    
    Returns:
        Dictionary berisi hasil eksperimen
    """
    global trainset, testset, client_indices, device
    
    # Setup
    print("\n" + "="*60)
    print("FEDERATED LEARNING SIMULATION")
    print("="*60)
    
    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load data
    print(f"\nLoading {dataset_name.upper()} dataset...")
    trainset, testset = load_dataset(dataset_name, data_path)
    
    # Partition data
    print(f"Partitioning data ({partition})...")
    if partition.lower() == 'iid':
        client_indices = partition_iid(trainset, num_clients, seed)
    else:
        client_indices = partition_dirichlet(trainset, num_clients, alpha, seed)
    
    # Print partition stats
    stats = get_partition_stats(trainset, client_indices)
    print(f"Samples per client: min={min(stats['samples_per_client'])}, "
          f"max={max(stats['samples_per_client'])}, "
          f"mean={np.mean(stats['samples_per_client']):.0f}")
    
    # Visualize partition if requested
    if visualize:
        viz_path = os.path.join(output_dir, f'partition_{timestamp}.png')
        print(f"Saving partition visualization to {viz_path}")
        visualize_partition(trainset, client_indices, viz_path)
    
    # Create initial model and parameters
    print("\nInitializing model...")
    if dataset_name.lower() == 'mnist':
        model = SimpleCNN(num_classes=10)
    else:
        model = CIFAR10CNN(num_classes=10)
    
    initial_parameters = ndarrays_to_parameters(get_parameters(model))
    
    # Configure strategy
    def fit_config(server_round: int) -> Dict:
        return {
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
            "server_round": server_round
        }
    
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit,
        min_fit_clients=max(2, int(num_clients * fraction_fit)),
        min_evaluate_clients=max(2, int(num_clients * fraction_fit)),
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config,
        initial_parameters=initial_parameters
    )
    
    # Run simulation
    print(f"\nStarting simulation with {num_clients} clients, {num_rounds} rounds...")
    print("-"*60)
    
    history = start_simulation(
        client_fn=get_client_fn(
            dataset_name, 
            batch_size, 
            local_epochs,
            learning_rate
        ),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0}
    )
    
    # Process results
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    
    # Extract metrics from history
    results = {
        'config': {
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'dataset': dataset_name,
            'partition': partition,
            'alpha': alpha if partition != 'iid' else None,
            'fraction_fit': fraction_fit,
            'local_epochs': local_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'seed': seed
        },
        'losses_distributed': [],
        'accuracies_distributed': [],
        'timestamp': timestamp
    }
    
    # Process distributed evaluation results
    if history.metrics_distributed:
        for round_num, metrics in history.metrics_distributed.get('accuracy', []):
            results['accuracies_distributed'].append({
                'round': round_num,
                'accuracy': metrics
            })
    
    # Process losses
    if history.losses_distributed:
        for round_num, loss in history.losses_distributed:
            results['losses_distributed'].append({
                'round': round_num,
                'loss': loss
            })
    
    # Print final results
    if results['accuracies_distributed']:
        final_acc = results['accuracies_distributed'][-1]['accuracy']
        print(f"Final accuracy: {final_acc*100:.2f}%")
    
    if results['losses_distributed']:
        final_loss = results['losses_distributed'][-1]['loss']
        print(f"Final loss: {final_loss:.4f}")
    
    # Save results
    results_path = os.path.join(output_dir, f'results_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Visualize results
    if visualize and results['accuracies_distributed']:
        plot_results(results, output_dir, timestamp)
    
    return results


def plot_results(results: Dict, output_dir: str, timestamp: str):
    """Plot hasil training."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    if results['losses_distributed']:
        rounds = [r['round'] for r in results['losses_distributed']]
        losses = [r['loss'] for r in results['losses_distributed']]
        axes[0].plot(rounds, losses, 'b-o', linewidth=2, markersize=6)
        axes[0].set_xlabel('Round', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training Loss per Round', fontsize=14)
        axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    if results['accuracies_distributed']:
        rounds = [r['round'] for r in results['accuracies_distributed']]
        accs = [r['accuracy'] * 100 for r in results['accuracies_distributed']]
        axes[1].plot(rounds, accs, 'g-o', linewidth=2, markersize=6)
        axes[1].set_xlabel('Round', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Test Accuracy per Round', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 100])
    
    plt.tight_layout()
    
    # Add configuration info
    config = results['config']
    fig.suptitle(
        f"{config['dataset'].upper()} | {config['num_clients']} clients | "
        f"{config['partition']} partition | {config['num_rounds']} rounds",
        y=1.02, fontsize=12
    )
    
    plot_path = os.path.join(output_dir, f'training_plot_{timestamp}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.show()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run Federated Learning Simulation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset and partitioning
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'cifar10'],
                        help='Dataset to use')
    parser.add_argument('--partition', type=str, default='iid',
                        choices=['iid', 'dirichlet'],
                        help='Data partition type')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet alpha (lower = more non-IID)')
    
    # FL configuration
    parser.add_argument('--num-clients', type=int, default=10,
                        help='Number of clients')
    parser.add_argument('--num-rounds', type=int, default=10,
                        help='Number of FL rounds')
    parser.add_argument('--fraction-fit', type=float, default=0.5,
                        help='Fraction of clients per round')
    parser.add_argument('--local-epochs', type=int, default=1,
                        help='Local epochs per round')
    
    # Training configuration
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Learning rate')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--data-path', type=str, default='./data',
                        help='Path for data storage')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Output directory')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Disable visualization')
    
    args = parser.parse_args()
    
    # Run simulation
    results = run_simulation(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        dataset_name=args.dataset,
        partition=args.partition,
        alpha=args.alpha,
        fraction_fit=args.fraction_fit,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        data_path=args.data_path,
        output_dir=args.output_dir,
        visualize=not args.no_visualize
    )
    
    print("\n✓ Simulation completed successfully!")


if __name__ == "__main__":
    main()
