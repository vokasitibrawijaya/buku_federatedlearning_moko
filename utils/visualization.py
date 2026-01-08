"""
visualization.py - Utilitas Visualisasi untuk Hasil FL

Modul ini menyediakan berbagai fungsi visualisasi untuk:
1. Training metrics (loss, accuracy)
2. Data partitioning
3. Client performance comparison
4. Convergence analysis

Author: Tim Buku Federated Learning
Date: Januari 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional


def plot_training_history(
    results: Dict,
    output_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot training loss dan accuracy dari hasil eksperimen.
    
    Args:
        results: Dictionary hasil eksperimen
        output_path: Path untuk menyimpan gambar
        show: Tampilkan plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Extract data
    if 'losses_distributed' in results:
        rounds = [r['round'] for r in results['losses_distributed']]
        losses = [r['loss'] for r in results['losses_distributed']]
        axes[0].plot(rounds, losses, 'b-o', linewidth=2, markersize=4)
        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].grid(True, alpha=0.3)
    
    if 'accuracies_distributed' in results:
        rounds = [r['round'] for r in results['accuracies_distributed']]
        accs = [r['accuracy'] * 100 for r in results['accuracies_distributed']]
        axes[1].plot(rounds, accs, 'g-o', linewidth=2, markersize=4)
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Test Accuracy')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 100])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_data_distribution(
    label_distribution: List[List[int]],
    num_classes: int = 10,
    output_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot distribusi label per klien.
    
    Args:
        label_distribution: List of label counts per client
        num_classes: Jumlah kelas
        output_path: Path untuk menyimpan gambar
        show: Tampilkan plot
    """
    num_clients = len(label_distribution)
    distribution = np.array(label_distribution)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap
    sns.heatmap(
        distribution.T,
        ax=axes[0],
        cmap='Blues',
        annot=True if num_clients <= 10 else False,
        fmt='d',
        xticklabels=[f'C{i}' for i in range(num_clients)],
        yticklabels=[f'Class {i}' for i in range(num_classes)]
    )
    axes[0].set_xlabel('Client')
    axes[0].set_ylabel('Class')
    axes[0].set_title('Label Distribution per Client')
    
    # Stacked bar chart
    x = np.arange(num_clients)
    bottom = np.zeros(num_clients)
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    for class_idx in range(num_classes):
        values = distribution[:, class_idx]
        axes[1].bar(x, values, bottom=bottom, color=colors[class_idx], 
                    label=f'Class {class_idx}')
        bottom += values
    
    axes[1].set_xlabel('Client')
    axes[1].set_ylabel('Number of Samples')
    axes[1].set_title('Sample Distribution per Client')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'C{i}' for i in range(num_clients)])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_convergence_comparison(
    results_list: List[Dict],
    labels: List[str],
    metric: str = 'accuracy',
    output_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Membandingkan konvergensi beberapa eksperimen.
    
    Args:
        results_list: List of result dictionaries
        labels: Label untuk setiap eksperimen
        metric: 'accuracy' atau 'loss'
        output_path: Path untuk menyimpan gambar
        show: Tampilkan plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
    
    for i, (results, label) in enumerate(zip(results_list, labels)):
        if metric == 'accuracy' and 'accuracies_distributed' in results:
            rounds = [r['round'] for r in results['accuracies_distributed']]
            values = [r['accuracy'] * 100 for r in results['accuracies_distributed']]
            ylabel = 'Accuracy (%)'
        elif metric == 'loss' and 'losses_distributed' in results:
            rounds = [r['round'] for r in results['losses_distributed']]
            values = [r['loss'] for r in results['losses_distributed']]
            ylabel = 'Loss'
        else:
            continue
        
        ax.plot(rounds, values, '-o', color=colors[i], label=label, 
                linewidth=2, markersize=4)
    
    ax.set_xlabel('Round')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{metric.capitalize()} Convergence Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if metric == 'accuracy':
        ax.set_ylim([0, 100])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_client_performance(
    client_metrics: Dict[int, Dict],
    output_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot performa per klien.
    
    Args:
        client_metrics: Dictionary {client_id: {metric: value}}
        output_path: Path untuk menyimpan gambar
        show: Tampilkan plot
    """
    client_ids = list(client_metrics.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy per client
    if 'accuracy' in client_metrics[client_ids[0]]:
        accuracies = [client_metrics[c]['accuracy'] * 100 for c in client_ids]
        colors = ['green' if a > np.mean(accuracies) else 'coral' for a in accuracies]
        axes[0].bar(client_ids, accuracies, color=colors)
        axes[0].axhline(y=np.mean(accuracies), color='red', linestyle='--', 
                        label=f'Mean: {np.mean(accuracies):.1f}%')
        axes[0].set_xlabel('Client ID')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title('Accuracy per Client')
        axes[0].legend()
    
    # Training time per client
    if 'training_time' in client_metrics[client_ids[0]]:
        times = [client_metrics[c]['training_time'] for c in client_ids]
        axes[1].bar(client_ids, times, color='steelblue')
        axes[1].axhline(y=np.mean(times), color='red', linestyle='--',
                        label=f'Mean: {np.mean(times):.1f}s')
        axes[1].set_xlabel('Client ID')
        axes[1].set_ylabel('Training Time (s)')
        axes[1].set_title('Training Time per Client')
        axes[1].legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_summary_report(
    results: Dict,
    output_path: str
) -> None:
    """
    Membuat summary report dalam format text.
    
    Args:
        results: Dictionary hasil eksperimen
        output_path: Path untuk menyimpan report
    """
    lines = []
    lines.append("=" * 60)
    lines.append("FEDERATED LEARNING EXPERIMENT SUMMARY")
    lines.append("=" * 60)
    lines.append("")
    
    # Configuration
    if 'config' in results:
        lines.append("CONFIGURATION")
        lines.append("-" * 40)
        for key, value in results['config'].items():
            lines.append(f"  {key}: {value}")
        lines.append("")
    
    # Final Results
    lines.append("FINAL RESULTS")
    lines.append("-" * 40)
    
    if 'accuracies_distributed' in results and results['accuracies_distributed']:
        final_acc = results['accuracies_distributed'][-1]['accuracy']
        lines.append(f"  Final Accuracy: {final_acc*100:.2f}%")
    
    if 'losses_distributed' in results and results['losses_distributed']:
        final_loss = results['losses_distributed'][-1]['loss']
        lines.append(f"  Final Loss: {final_loss:.4f}")
    
    lines.append("")
    lines.append("=" * 60)
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Summary report saved to {output_path}")


def load_results(filepath: str) -> Dict:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Visualize FL Results')
    
    parser.add_argument(
        '--results', type=str, required=True,
        help='Path to results JSON file'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./plots',
        help='Directory for output plots'
    )
    parser.add_argument(
        '--no-show', action='store_true',
        help='Do not display plots (just save)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results = load_results(args.results)
    
    # Generate plots
    plot_training_history(
        results,
        output_path=str(output_dir / 'training_history.png'),
        show=not args.no_show
    )
    
    # Generate summary
    create_summary_report(
        results,
        output_path=str(output_dir / 'summary.txt')
    )
    
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()
