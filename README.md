# 🌐 Federated Learning Book - Practical Simulations

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flower](https://img.shields.io/badge/Flower-1.6.0-green.svg)](https://flower.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Repository Kode Pendamping Buku "Federated Learning: Teori dan Praktik"**

Repository ini berisi kode simulasi Federated Learning yang **siap dijalankan** dan telah **diuji**. Semua contoh kode di Bab 22 dapat dipraktikkan langsung menggunakan kode di repository ini.

---

## 📚 Daftar Isi

- [Instalasi](#-instalasi)
- [Struktur Folder](#-struktur-folder)
- [Quick Start](#-quick-start)
- [Simulasi Dasar](#-simulasi-dasar)
- [Simulasi Heterogenitas](#-simulasi-heterogenitas)
- [Studi Kasus Indonesia](#-studi-kasus-indonesia)
- [Docker Deployment](#-docker-deployment)
- [Troubleshooting](#-troubleshooting)

---

## 🚀 Instalasi

### Prasyarat
- Python 3.9 atau lebih baru
- pip atau conda
- (Opsional) Docker dan Docker Compose
- (Opsional) GPU dengan CUDA untuk training lebih cepat

### Opsi 1: Menggunakan pip (Recommended)

```bash
# Clone repository
git clone https://github.com/atmoko-lab/fl-book-code.git
cd fl-book-code

# Buat virtual environment
python -m venv fl_env

# Aktivasi (Windows)
fl_env\Scripts\activate

# Aktivasi (Linux/Mac)
source fl_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Opsi 2: Menggunakan Conda

```bash
# Dari file environment.yml
conda env create -f environment.yml
conda activate fl_book

# Verifikasi instalasi
python -c "import flwr; print(f'Flower version: {flwr.__version__}')"
```

### Opsi 3: Menggunakan Docker

```bash
# Build images
docker-compose build

# Jalankan simulasi
docker-compose up
```

---

## 📁 Struktur Folder

```
fl-book-code/
├── basic/                    # Simulasi FL dasar
│   ├── model.py             # Definisi model CNN/MLP
│   ├── data.py              # Data loading dan partisi
│   ├── train.py             # Training dan evaluasi
│   ├── client.py            # Flower client
│   ├── server.py            # Flower server
│   └── run_simulation.py    # Script simulasi all-in-one
│
├── heterogeneous/           # Simulasi dengan heterogenitas
│   ├── client_heterogeneous.py
│   ├── server_heterogeneous.py
│   └── configs/
│
├── case_studies/            # Studi kasus Indonesia
│   ├── hospital_xray/       # Deteksi penyakit RS
│   ├── fintech_fraud/       # Deteksi fraud
│   └── manufacturing_pdm/   # Predictive maintenance
│
├── docker/                  # Docker configuration
│   ├── Dockerfile.server
│   ├── Dockerfile.client
│   └── docker-compose.yml
│
├── configs/                 # Konfigurasi eksperimen
│   ├── basic_mnist.yaml
│   ├── noniid_cifar.yaml
│   └── ...
│
├── utils/                   # Utilities
│   ├── visualization.py
│   ├── partitioning.py
│   └── metrics.py
│
├── tests/                   # Unit tests
│   └── test_simulation.py
│
├── requirements.txt         # Dependencies
├── environment.yml          # Conda environment
└── README.md               # Dokumentasi ini
```

---

## ⚡ Quick Start

### Menjalankan Simulasi Pertama (5 menit)

```bash
# Masuk ke folder basic
cd basic

# Jalankan simulasi lengkap (dalam satu script)
python run_simulation.py --num-clients 5 --num-rounds 10

# Output akan menampilkan:
# - Progress training per ronde
# - Akurasi dan loss per ronde
# - Plot hasil di akhir
```

### Menjalankan dengan Multiple Terminal

```bash
# Terminal 1: Jalankan server
python server.py

# Terminal 2: Jalankan client 0
python client.py --client-id 0

# Terminal 3: Jalankan client 1
python client.py --client-id 1

# ... tambahkan client sesuai kebutuhan
```

---

## 📊 Simulasi Dasar

### MNIST dengan IID Data

```bash
cd basic
python run_simulation.py \
    --dataset mnist \
    --num-clients 10 \
    --num-rounds 20 \
    --partition iid
```

**Expected Output:**
- Ronde 5: ~92% accuracy
- Ronde 10: ~96% accuracy
- Ronde 20: ~98% accuracy

### CIFAR-10 dengan Non-IID Data

```bash
python run_simulation.py \
    --dataset cifar10 \
    --num-clients 10 \
    --num-rounds 30 \
    --partition dirichlet \
    --alpha 0.5
```

### Konfigurasi Custom (YAML)

```bash
python run_simulation.py --config ../configs/custom_experiment.yaml
```

---

## 🔀 Simulasi Heterogenitas

### System Heterogeneity

Simulasi dengan klien yang memiliki kemampuan berbeda:

```bash
cd heterogeneous
python run_heterogeneous.py \
    --slow-clients 3 \
    --fast-clients 7 \
    --dropout-rate 0.1
```

### Dengan Docker (Resource Limiting)

```bash
cd docker
docker-compose -f docker-compose-heterogeneous.yml up
```

---

## 🇮🇩 Studi Kasus Indonesia

### Studi Kasus 1: Kolaborasi Rumah Sakit

Simulasi 5 RS berkolaborasi untuk deteksi kanker paru:

```bash
cd case_studies/hospital_xray
python run_hospital_fl.py
```

### Studi Kasus 2: Deteksi Fraud Fintech

Simulasi 10 fintech untuk deteksi fraud:

```bash
cd case_studies/fintech_fraud
python run_fintech_fl.py
```

### Studi Kasus 3: Predictive Maintenance

Simulasi 8 pabrik untuk prediksi kerusakan:

```bash
cd case_studies/manufacturing_pdm
python run_manufacturing_fl.py
```

---

## 🐳 Docker Deployment

### Build dan Run

```bash
cd docker

# Build images
docker-compose build

# Run dengan 5 clients
docker-compose up --scale client=5

# Stop
docker-compose down
```

### Custom Resource Limits

Edit `docker-compose.yml` untuk mengatur CPU/memory per container.

---

## 🔧 Troubleshooting

### "Connection refused" Error

```bash
# Pastikan server sudah running
# Cek port tidak diblokir
netstat -an | findstr 8080  # Windows
netstat -an | grep 8080     # Linux/Mac
```

### "CUDA out of memory"

```bash
# Kurangi batch size
python run_simulation.py --batch-size 16

# Atau gunakan CPU
python run_simulation.py --device cpu
```

### Import Error

```bash
# Pastikan environment aktif
pip list | grep flwr

# Reinstall jika perlu
pip install --upgrade -r requirements.txt
```

---

## 📈 Visualisasi Hasil

```bash
cd utils
python visualization.py --results ../results/experiment_001.json
```

Output:
- `training_loss.png` - Loss per ronde
- `accuracy.png` - Akurasi per ronde
- `data_distribution.png` - Distribusi data per klien

---

## 📝 Konfigurasi Referensi

Lihat folder `configs/` untuk contoh konfigurasi lengkap:

| File | Deskripsi |
|------|-----------|
| `basic_mnist.yaml` | MNIST IID, 10 klien |
| `noniid_cifar.yaml` | CIFAR-10 non-IID |
| `hospital.yaml` | Studi kasus RS |
| `fintech.yaml` | Studi kasus fintech |

---

## 🤝 Kontribusi

Pull requests welcome! Untuk perubahan besar, silakan buka issue terlebih dahulu.

---

## 📜 Lisensi

MIT License - Silakan gunakan untuk keperluan akademik dan komersial.

---

## 📖 Sitasi

Jika menggunakan kode ini untuk penelitian, silakan sitasi:

```bibtex
@book{fl_book_2026,
    title={Federated Learning: Teori dan Praktik},
    author={Tim Penulis},
    year={2026},
    publisher={Penerbit}
}
```

---

## 📞 Kontak

- Email: federated.learning.book@example.com
- GitHub Issues: [Link](https://github.com/atmoko-lab/fl-book-code/issues)
