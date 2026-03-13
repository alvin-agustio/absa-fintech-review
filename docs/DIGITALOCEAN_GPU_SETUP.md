# DigitalOcean GPU Droplet Setup

Panduan ini ditujukan untuk menjalankan repo skripsi ini di GPU Droplet DigitalOcean dengan asumsi:

- OS/image: DigitalOcean AI/ML-ready NVIDIA image
- Python: 3.10+
- Workflow utama: training `train_lora.py`, `train_baseline.py`, `retrain_filtered.py`, dan optional `app.py`

## 1. Buat GPU Droplet

Pilih GPU Droplet DigitalOcean dan gunakan image AI/ML-ready NVIDIA agar driver GPU dan software dasarnya sudah terpasang.

Referensi resmi:

- DigitalOcean GPU Droplets: <https://docs.digitalocean.com/products/gpu-droplets/>
- Recommended GPU setup: <https://docs.digitalocean.com/products/droplets/getting-started/recommended-gpu-setup/>

## 2. Clone repo ke droplet

```bash
git clone <REPO_URL> skripsi
cd skripsi
```

Kalau repo belum memakai Git remote, kamu bisa upload folder ini via `scp` atau `rsync`.

## 3. Jalankan script setup

Script ini akan:

- cek GPU dengan `nvidia-smi`
- membuat virtual environment `.venv`
- menginstal PyTorch CUDA wheel
- menginstal dependency proyek

```bash
chmod +x scripts/setup_digitalocean_gpu.sh
./scripts/setup_digitalocean_gpu.sh
```

Secara default script memakai CUDA wheel PyTorch `cu124` dari index resmi PyTorch. Jika image kamu memerlukan wheel lain, override `TORCH_INDEX_URL`.

Contoh:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 ./scripts/setup_digitalocean_gpu.sh
```

Referensi resmi PyTorch:

- <https://docs.pytorch.org/get-started/locally/>

## 4. Aktivasi environment

```bash
source .venv/bin/activate
```

## 5. Verifikasi GPU dari Python

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

Kalau output `True` dan nama GPU muncul, environment sudah siap.

## 6. Jalankan training

LoRA:

```bash
python train_lora.py
```

Baseline:

```bash
python train_baseline.py
```

Retrain filtered:

```bash
python retrain_filtered.py
```

## 7. Jalankan evaluasi

```bash
python evaluate.py
```

## 8. Jalankan dashboard Streamlit

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Lalu buka:

```text
http://YOUR_DROPLET_IP:8501
```

Pastikan firewall DigitalOcean dan firewall OS mengizinkan port `8501` jika dashboard ingin diakses dari browser lokal.

## 9. Rekomendasi operasional

- Gunakan `tmux` atau `screen` untuk training panjang.
- Simpan model hasil training ke volume atau snapshot kalau hasilnya penting.
- Untuk training batch besar, mulai dari `train_lora.py` karena paling hemat parameter.

Contoh pakai `tmux`:

```bash
tmux new -s skripsi-train
source .venv/bin/activate
python train_lora.py
```

## 10. Troubleshooting singkat

`nvidia-smi` tidak ada:
- Kamu kemungkinan memakai image Ubuntu biasa. Pakai AI/ML-ready NVIDIA image atau install driver NVIDIA dulu.

`torch.cuda.is_available()` bernilai `False`:
- Biasanya wheel PyTorch tidak cocok dengan image CUDA/driver yang aktif. Coba ganti `TORCH_INDEX_URL` ke wheel resmi lain dari selector PyTorch.

`ModuleNotFoundError: peft` atau `streamlit`:
- Jalankan ulang setup script atau `pip install -r requirements.txt`.
