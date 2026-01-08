"""
run_hospital_fl.py - Simulasi FL untuk Kolaborasi Rumah Sakit Indonesia

Studi Kasus: 5 RS besar di Jawa berkolaborasi untuk deteksi penyakit
tanpa membagikan data pasien.

Konteks:
- RS Rujukan: RSUP Dr. Sardjito (Jogja), RSUP Dr. Kariadi (Semarang),
  RSUD Dr. Soetomo (Surabaya), RS Hasan Sadikin (Bandung), RSUP Fatmawati (Jakarta)
- Regulasi: UU No. 17 Tahun 2023 tentang Kesehatan
- Masalah: Data tidak bisa dikumpulkan secara terpusat

Simulasi ini menggunakan dataset MNIST sebagai proxy untuk citra X-ray
(untuk tujuan demonstrasi - di produksi gunakan dataset medis sebenarnya)

Author: Tim Buku Federated Learning
Date: Januari 2026
"""

import flwr as fl
from flwr.common import ndarrays_to_parameters, Metrics
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basic.model import SimpleCNN, get_parameters, set_parameters
from basic.data import load_dataset, get_client_dataloader
from basic.train import train, test


# Konfigurasi 5 RS Indonesia
HOSPITAL_CONFIG = {
    0: {
        "name": "RSUP Dr. Sardjito (Yogyakarta)",
        "size": 10000,  # Jumlah sampel
        "positive_ratio": 0.30,  # RS rujukan, banyak kasus positif
        "type": "rujukan"
    },
    1: {
        "name": "RSUP Dr. Kariadi (Semarang)",
        "size": 8000,
        "positive_ratio": 0.25,
        "type": "rujukan"
    },
    2: {
        "name": "RSUD Dr. Soetomo (Surabaya)",
        "size": 5000,
        "positive_ratio": 0.10,
        "type": "umum"
    },
    3: {
        "name": "RS Hasan Sadikin (Bandung)",
        "size": 3000,
        "positive_ratio": 0.08,
        "type": "umum"
    },
    4: {
        "name": "RSUP Fatmawati (Jakarta)",
        "size": 2000,
        "positive_ratio": 0.05,
        "type": "umum"
    }
}


def create_hospital_partition(
    dataset,
    hospital_config: Dict,
    seed: int = 42
) -> List[List[int]]:
    """
    Membuat partisi data yang mensimulasikan distribusi RS Indonesia.
    
    Karakteristik:
    - Quantity skew: RS besar punya lebih banyak data
    - Label skew: RS rujukan punya lebih banyak kasus positif
    
    Args:
        dataset: PyTorch dataset
        hospital_config: Konfigurasi per RS
        seed: Random seed
    
    Returns:
        List of indices untuk setiap RS
    """
    np.random.seed(seed)
    
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    
    # Untuk simulasi, kita treat class 0-4 sebagai "normal" dan 5-9 sebagai "abnormal"
    # (Ini hanya untuk demonstrasi - di produksi gunakan dataset medis sebenarnya)
    normal_classes = [0, 1, 2, 3, 4]
    abnormal_classes = [5, 6, 7, 8, 9]
    
    normal_indices = np.where(np.isin(labels, normal_classes))[0]
    abnormal_indices = np.where(np.isin(labels, abnormal_classes))[0]
    
    np.random.shuffle(normal_indices)
    np.random.shuffle(abnormal_indices)
    
    hospital_indices = []
    normal_ptr = 0
    abnormal_ptr = 0
    
    for hospital_id in range(len(hospital_config)):
        config = hospital_config[hospital_id]
        target_size = config["size"]
        positive_ratio = config["positive_ratio"]
        
        # Hitung jumlah normal dan abnormal
        num_abnormal = int(target_size * positive_ratio)
        num_normal = target_size - num_abnormal
        
        # Ambil indices
        hospital_normal = normal_indices[normal_ptr:normal_ptr + num_normal].tolist()
        hospital_abnormal = abnormal_indices[abnormal_ptr:abnormal_ptr + num_abnormal].tolist()
        
        normal_ptr += num_normal
        abnormal_ptr += num_abnormal
        
        # Combine dan shuffle
        hospital_data = hospital_normal + hospital_abnormal
        np.random.shuffle(hospital_data)
        
        hospital_indices.append(hospital_data)
        
        print(f"RS {hospital_id} ({config['name'][:20]}...): "
              f"{len(hospital_data)} samples, "
              f"{num_abnormal} positif ({positive_ratio*100:.0f}%)")
    
    return hospital_indices


# Global variables
trainset = None
testset = None
hospital_indices = None
device = None


def get_hospital_client_fn(batch_size: int, local_epochs: int, learning_rate: float):
    """Factory untuk client_fn RS."""
    global trainset, testset, hospital_indices, device
    
    class HospitalClient(fl.client.NumPyClient):
        def __init__(self, cid: str):
            self.cid = int(cid)
            self.hospital_name = HOSPITAL_CONFIG[self.cid]["name"]
            
            self.model = SimpleCNN(num_classes=10)
            self.model.to(device)
            
            self.trainloader = get_client_dataloader(
                trainset,
                hospital_indices[self.cid],
                batch_size
            )
            self.testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        
        def get_parameters(self, config):
            return get_parameters(self.model)
        
        def fit(self, parameters, config):
            set_parameters(self.model, parameters)
            
            history = train(
                self.model,
                self.trainloader,
                epochs=local_epochs,
                device=device,
                learning_rate=learning_rate
            )
            
            return (
                get_parameters(self.model),
                len(self.trainloader.dataset),
                {
                    "loss": history['loss'][-1],
                    "hospital_id": self.cid,
                    "hospital_name": self.hospital_name
                }
            )
        
        def evaluate(self, parameters, config):
            set_parameters(self.model, parameters)
            loss, accuracy = test(self.model, self.testloader, device)
            return (
                float(loss),
                len(self.testloader.dataset),
                {"accuracy": accuracy, "hospital_id": self.cid}
            )
    
    def client_fn(cid: str) -> fl.client.NumPyClient:
        return HospitalClient(cid)
    
    return client_fn


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Weighted average."""
    if not metrics:
        return {}
    
    total_examples = sum(n for n, _ in metrics)
    result = {}
    
    if 'accuracy' in metrics[0][1]:
        acc = sum(n * m['accuracy'] for n, m in metrics) / total_examples
        result['accuracy'] = acc
    
    return result


def run_hospital_simulation(
    num_rounds: int = 50,
    fraction_fit: float = 1.0,  # Semua RS berpartisipasi setiap ronde
    local_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    seed: int = 42,
    output_dir: str = './results_hospital'
) -> Dict:
    """
    Jalankan simulasi FL untuk kolaborasi RS Indonesia.
    
    Args:
        num_rounds: Jumlah ronde FL
        fraction_fit: Fraksi RS per ronde
        local_epochs: Epoch lokal per ronde
        batch_size: Ukuran batch
        learning_rate: Learning rate
        seed: Random seed
        output_dir: Directory output
    
    Returns:
        Dictionary hasil simulasi
    """
    global trainset, testset, hospital_indices, device
    
    print("\n" + "="*70)
    print("SIMULASI FL: KOLABORASI RUMAH SAKIT INDONESIA")
    print("Deteksi Penyakit dari Citra Medis")
    print("="*70)
    
    # Setup
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load data (MNIST sebagai proxy)
    print("\nMemuat dataset...")
    print("(Catatan: Menggunakan MNIST sebagai proxy untuk demonstrasi)")
    print("(Di produksi: Gunakan dataset medis seperti ChestX-ray14)")
    trainset, testset = load_dataset('mnist', './data')
    
    # Partition sesuai konfigurasi RS
    print("\nMembuat partisi data sesuai karakteristik RS...")
    hospital_indices = create_hospital_partition(trainset, HOSPITAL_CONFIG, seed)
    
    # Initialize model
    model = SimpleCNN(num_classes=10)
    initial_parameters = ndarrays_to_parameters(get_parameters(model))
    
    # Strategy - menggunakan FedAvg
    def fit_config(server_round: int) -> Dict:
        return {
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
            "server_round": server_round
        }
    
    num_hospitals = len(HOSPITAL_CONFIG)
    
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit,
        min_fit_clients=num_hospitals,
        min_evaluate_clients=num_hospitals,
        min_available_clients=num_hospitals,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config,
        initial_parameters=initial_parameters
    )
    
    # Run simulation
    print(f"\nMemulai simulasi dengan {num_hospitals} RS, {num_rounds} ronde...")
    print("-"*70)
    
    history = start_simulation(
        client_fn=get_hospital_client_fn(batch_size, local_epochs, learning_rate),
        num_clients=num_hospitals,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0}
    )
    
    # Process results
    print("\n" + "="*70)
    print("SIMULASI SELESAI")
    print("="*70)
    
    results = {
        'config': {
            'scenario': 'hospital_collaboration',
            'num_hospitals': num_hospitals,
            'num_rounds': num_rounds,
            'local_epochs': local_epochs,
            'learning_rate': learning_rate,
            'seed': seed
        },
        'hospitals': HOSPITAL_CONFIG,
        'accuracies': [],
        'losses': [],
        'timestamp': timestamp
    }
    
    # Extract metrics
    if history.metrics_distributed:
        for round_num, metrics in history.metrics_distributed.get('accuracy', []):
            results['accuracies'].append({'round': round_num, 'accuracy': metrics})
    
    if history.losses_distributed:
        for round_num, loss in history.losses_distributed:
            results['losses'].append({'round': round_num, 'loss': loss})
    
    # Analysis
    print("\n" + "-"*70)
    print("ANALISIS HASIL")
    print("-"*70)
    
    if results['accuracies']:
        final_acc = results['accuracies'][-1]['accuracy']
        print(f"\nAkurasi global akhir: {final_acc*100:.2f}%")
        
        # Expected results
        print("\nPerbandingan dengan skenario lain (estimasi):")
        print(f"  - Model lokal per RS (tanpa FL): ~70-80%")
        print(f"  - Federated Learning (hasil ini): {final_acc*100:.1f}%")
        print(f"  - Data terpusat (jika diizinkan): ~95%+")
        
        print("\nManfaat FL untuk RS:")
        print("  ✓ Data tetap di masing-masing RS")
        print("  ✓ Privasi pasien terjaga")
        print("  ✓ RS kecil mendapat benefit dari RS besar")
        print("  ✓ Memenuhi regulasi UU Kesehatan")
    
    # Save results
    results_path = os.path.join(output_dir, f'results_hospital_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nHasil disimpan di: {results_path}")
    
    # Plot
    plot_hospital_results(results, output_dir, timestamp)
    
    return results


def plot_hospital_results(results: Dict, output_dir: str, timestamp: str):
    """Plot hasil simulasi RS."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss plot
    if results['losses']:
        rounds = [r['round'] for r in results['losses']]
        losses = [r['loss'] for r in results['losses']]
        axes[0].plot(rounds, losses, 'b-', linewidth=2)
        axes[0].set_xlabel('Ronde')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    if results['accuracies']:
        rounds = [r['round'] for r in results['accuracies']]
        accs = [r['accuracy'] * 100 for r in results['accuracies']]
        axes[1].plot(rounds, accs, 'g-', linewidth=2)
        axes[1].set_xlabel('Ronde')
        axes[1].set_ylabel('Akurasi (%)')
        axes[1].set_title('Test Accuracy')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 100])
    
    # Hospital data distribution
    hospitals = results['hospitals']
    names = [f"RS{i}" for i in range(len(hospitals))]
    sizes = [hospitals[str(i)]["size"] for i in range(len(hospitals))]
    positive_ratios = [hospitals[str(i)]["positive_ratio"] * 100 for i in range(len(hospitals))]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = axes[2].bar(x - width/2, np.array(sizes)/1000, width, label='Jumlah Data (ribu)', color='steelblue')
    axes[2].set_ylabel('Jumlah Data (ribu)')
    axes[2].set_xlabel('Rumah Sakit')
    axes[2].set_title('Distribusi Data per RS')
    
    ax2 = axes[2].twinx()
    bars2 = ax2.bar(x + width/2, positive_ratios, width, label='Rasio Positif (%)', color='coral')
    ax2.set_ylabel('Rasio Positif (%)')
    
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names)
    axes[2].legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    fig.suptitle(
        "Simulasi FL: Kolaborasi 5 RS Indonesia untuk Deteksi Penyakit",
        y=1.02, fontsize=12, fontweight='bold'
    )
    
    plot_path = os.path.join(output_dir, f'plot_hospital_{timestamp}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot disimpan di: {plot_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Simulasi FL Kolaborasi RS Indonesia',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--num-rounds', type=int, default=50)
    parser.add_argument('--local-epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='./results_hospital')
    
    args = parser.parse_args()
    
    run_hospital_simulation(
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    print("\n✓ Simulasi selesai!")


if __name__ == "__main__":
    main()
