# SegFormer 7-Band — Land Cover Segmentation Pipeline (versi AMD / ROCm)

> **Versi ini untuk GPU AMD.** Diadaptasi dari `vale-segformer-7band` (NVIDIA/CUDA)
> untuk Windows 11 + **Python 3.12** + **Radeon RX 9070 16 GB (RDNA4, gfx1201)**.
>
> **Instalasi cepat (CMD):**
> ```bat
> setup_venv.bat              :: venv Python 3.12 + PyTorch ROCm + requirements
> copy_data_from_cuda.bat     :: (opsional) salin data/ & best_model.pth dari versi CUDA
> venv\Scripts\activate
> python gpu_utils.py         :: harus menampilkan "AMD Radeon RX 9070 ... ROCm/HIP"
> ```
>
> **Yang berubah dibanding versi CUDA:**
> | Bagian | CUDA (lama) | AMD (ini) |
> |---|---|---|
> | Python | 3.11 | 3.12 (wheel ROCm Windows cp312) |
> | PyTorch | `download.pytorch.org/whl/cu128` | repo AMD ROCm (`setup_venv.bat`) |
> | Pilih GPU | selalu `cuda:0` | otomatis: iGPU Ryzen 7700 dilewati, pilih GPU `GPU_ARCH = "gfx1201"` (RX 9070); paksa lewat `GPU_INDEX` |
> | AMP | float16 + GradScaler | **bfloat16** tanpa GradScaler (`AMP_DTYPE`, bisa balik ke `"float16"`) |
> | `BATCH_SIZE` | 2 (RTX 3050 6 GB) | 8 (16 GB VRAM) |
> | `INFERENCE_BATCH_SIZE` | 4 | 16 |
> | `NUM_WORKERS` | 2 | 4 (RAM 32 GB) |
> | Helper GPU | tersebar di script | `gpu_utils.py` (device, AMP, env MIOpen, self-test) |
> | Resume checkpoint | `torch.load` default | `weights_only=False` (aman untuk PyTorch ≥ 2.6) |
>
> Di PyTorch ROCm, nama device **tetap `"cuda"`** (API `torch.cuda` dipetakan ke HIP),
> jadi `DEVICE = "cuda"` di `config.py` = GPU AMD. Checkpoint `.pth` dari versi CUDA
> bisa langsung dipakai untuk inference / resume di versi ini.
>
> **Troubleshooting AMD:**
> - `torch.cuda.is_available() == False` → update driver Adrenalin, hapus AMD HIP SDK lama, restart.
> - `hipErrorInvalidImage` / "invalid device kernel image" → wheel bukan untuk gfx1201; ulangi `setup_venv.bat` (hapus folder `venv` dulu).
> - `HIP out of memory` → turunkan `BATCH_SIZE` (6 → 4) / `INFERENCE_BATCH_SIZE`.
> - Epoch pertama lambat → MIOpen mencari kernel konvolusi; sudah di-set `MIOPEN_FIND_MODE=FAST` di `gpu_utils.py`.
> - Jangan set `HSA_OVERRIDE_GFX_VERSION` (tidak perlu untuk RX 9070).


Pipeline segmentasi tutupan lahan dari citra orthophoto RGB memakai
[SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer)
yang diperluas dari 3 channel (RGB) menjadi **7 band**: `R, G, B, ExG, VARI,
Edge, Texture`. Termasuk tahap enhancement citra (CLAHE), preprocessing,
training, inference ke SHP siap-QGIS, dan uji akurasi lapangan.

## Struktur folder

```
vale-segformer-7band-amd/
├── config.py                          # konfigurasi terpusat (path, hyperparameter, dsb)
├── gpu_utils.py                       # helper GPU AMD/ROCm (device, AMP bf16, self-test)
├── requirements.txt
├── setup_venv.bat                     # buat venv Python 3.12 + PyTorch ROCm
├── data_qc/                           # quality-control data (jalankan sesuai kebutuhan)
│   ├── check_jpg_integrity.py         # cek JPG corrupt di folder foto lapangan/drone
│   ├── check_tif_metadata.py          # cek CRS/resolusi 1 file GeoTIFF
│   ├── validate_mask_data.py          # validasi & perbaiki mask .npy hasil preprocessing
│   ├── analyze_class_distribution.py  # distribusi kelas train/val + rekomendasi weights
│   └── calculate_class_weights.py     # hitung class weights (dipanggil otomatis saat training)
├── preprocessing/                     # TAHAP 1 & 2
│   ├── clahe_enhancement.py           # (1) samakan tone semua site ke 1 site referensi
│   └── build_patches.py               # (2) TIF+SHP -> citra 7-band -> patch .npy
├── training/                          # TAHAP 3
│   └── train_model.py                 # training SegFormer 7-band
├── inference/                         # TAHAP 4
│   ├── run_inference.py               # resample -> prediksi (tile + margin konteks) -> SHP -> style QGIS
│   ├── fix_tile_artifacts.py          # post-process: perbaiki blob kotak salah-kelas 1 tile (dipanggil otomatis)
│   ├── filter_small_polygons.py       # post-process: sieve poligon kecil (dipanggil otomatis)
│   └── smooth_polygon_boundaries.py   # post-process: smoothing batas poligon (dipanggil otomatis)
├── accuracy_assessment/               # TAHAP 5 (opsional, uji akurasi lapangan)
│   ├── grid_sampling_assessment.py    # sampling grid representatif dari nol (Cochran+FPC)
│   └── final_grid_assessment.py       # hitung metrik pakai grid final yang sudah ada
└── data/                              # (dibuat otomatis) dataset kerja — lihat di bawah
    ├── raw/<site>/*.tif + *.shp       # input mentah, per site
    └── processed/
        ├── clahe_enhanced/<site>/     # output preprocessing/clahe_enhancement.py
        ├── patches/<site>/*.npy       # output preprocessing/build_patches.py
        └── masks/<site>/*.npy
```

Semua path di atas (kecuali `config.py`) diambil dari `Config` di
`config.py` — **ubah path di sana**, bukan di masing-masing script.

## Alur pipeline (urutan menjalankan)

```
1. data_qc/check_jpg_integrity.py      (opsional, cek foto mentah)
   data_qc/check_tif_metadata.py       (opsional, cek CRS/resolusi TIF)
        │
        ▼
2. preprocessing/clahe_enhancement.py  → data/processed/clahe_enhanced/<site>/
        │
        ▼
3. preprocessing/build_patches.py      → data/processed/patches|masks/<site>/*.npy
        │
        ▼
4. data_qc/validate_mask_data.py           (opsional, cek mask corrupt/invalid)
   data_qc/analyze_class_distribution.py   (opsional, cek imbalance train/val)
   data_qc/calculate_class_weights.py      (opsional, lihat class weights train)
        │  → CLASS_WEIGHTS = "auto": dihitung otomatis saat training
        ▼
5. training/train_model.py             → outputs/models/best_model.pth
        │
        ▼
6. inference/run_inference.py          → outputs/job_<n>/<nama>.shp (+ .qml siap QGIS)
        │
        ▼
7. accuracy_assessment/grid_sampling_assessment.py   (opsional, uji akurasi lapangan)
   accuracy_assessment/final_grid_assessment.py      (opsional, jika grid sampel sudah ada)
```

### 1. Setup awal

```bat
setup_venv.bat
venv\Scripts\activate
python gpu_utils.py
```

(Jangan `pip install -r requirements.txt` langsung tanpa PyTorch ROCm — pip
akan memasang torch CPU dari PyPI. `setup_venv.bat` mengurus urutannya.)

Susun data mentah di `data/raw/<nama_site>/`, masing-masing berisi satu
`.tif` (citra RGB) dan satu `.shp` (label kelas, kolom kelasnya diatur via
`Config.KELAS_COL`, default `"ID_NEW"`, berisi ID kelas 0–9).

Cek struktur data & konfigurasi kapan saja dengan:

```bash
python -c "from config import Config; Config.diagnose()"
```

### 2. CLAHE enhancement (menyamakan tone antar site)

```bash
python preprocessing/clahe_enhancement.py
```

Memilih satu site sebagai referensi (otomatis, atau set
`Config.CLAHE_REF_SITE`), mengoptimalkan `clipLimit` CLAHE dengan
memaksimalkan jarak Jeffries-Matusita antar `Config.CONFUSING_PAIRS`, lalu
menyamakan histogram semua site lain ke referensi tersebut (tile-wise
histogram matching). Output: `data/processed/clahe_enhanced/<site>/`.

### 3. Build patch 7-band + mask

```bash
python preprocessing/build_patches.py
```

Untuk tiap site: RGB → 7 band (`R, G, B, ExG, VARI, Edge, Texture`) →
normalisasi → rasterize SHP jadi mask → potong jadi patch
`Config.PATCH_SIZE` (default 512) dengan overlap `Config.OVERLAP`. Patch
disimpan sebagai `.npy` di `data/processed/patches/` & `masks/`.

### 4. QA dataset (opsional tapi disarankan sebelum training)

```bash
python data_qc/validate_mask_data.py
python data_qc/analyze_class_distribution.py
```

`analyze_class_distribution.py` mem-plot perbandingan distribusi kelas
train vs val. Class weights **tidak perlu disalin manual**: dengan
`CLASS_WEIGHTS = "auto"` (default) bobot dihitung otomatis dari mask split
train saat training dimulai, memakai `CLASS_WEIGHT_METHOD` (`"balanced"`,
`"sqrt"`, `"effective"`, `"sklearn"`). Hasilnya di-cache di
`outputs/logs/class_weights.json` dan hanya dihitung ulang kalau dataset
berubah. Set `CLASS_WEIGHTS = None` untuk menonaktifkan, atau isi list
untuk override manual. Untuk melihat bobotnya tanpa training:
`python data_qc/calculate_class_weights.py`.

### 5. Training

```bash
python training/train_model.py
```

Checkpoint periodik disimpan di `outputs/checkpoints/`, model terbaik di
`outputs/models/best_model.pth` (dipilih berdasarkan `BEST_MODEL_METRIC`:
`"miou"` atau `"fwiou"`). Plot loss/IoU/akurasi per epoch tersimpan di
`outputs/logs/training_plots.png`.

Metrik dihitung dari confusion matrix yang diakumulasi selama satu epoch
penuh (bukan rata-rata per batch):

| Metrik | Arti |
|---|---|
| **mIoU** | rata-rata IoU per kelas; kelas yang tidak ada di GT & prediksi diabaikan |
| **FWIoU** | frequency-weighted IoU: IoU tiap kelas dibobot proporsi pixel GT-nya |
| **Overall Acc** | pixel accuracy |

Setiap kali best model baru tercapai, tabel IoU per kelas dicetak dan
detailnya (termasuk confusion matrix & class weights) disimpan ke
`outputs/logs/best_metrics.json` — siap dipakai untuk laporan.

### 6. Inference

```bash
python inference/run_inference.py --input path/ke/citra.tif --resolution 0.1
python inference/run_inference.py --input folder_citra/ --model outputs/models/best_model.pth
```

`python predict.py ...` dengan argumen yang sama juga bisa dipakai (pintasan
ke script yang sama). Alurnya per file TIF: resample (opsional; TIF lat/lon
otomatis direproject ke UTM) → prediksi per tile → vektorisasi ke SHP →
`filter_small_polygons.py` (buang poligon noise) →
`smooth_polygon_boundaries.py` (haluskan batas) → `.qml` (style QGIS sesuai
`Config.CLASS_COLORS`/`CLASS_NAMES`). Output per run ada di `outputs/job_<n>/`.

Cara prediksi per tile:

- **Margin konteks**: tiap tile (default 2048 px) dibaca dengan tambahan
  `tile_overlap / 2` px di setiap sisi, supaya band Edge & Texture di tepi
  tile dihitung sama seperti saat training (dari citra utuh), lalu margin
  dibuang lagi.
- **Blending probabilitas**: patch 512 px di dalam tile (overlap 75%) dan
  tile di dalam raster sama-sama di-blend dengan bobot Gaussian di level
  probabilitas softmax, baru diambil argmax di akhir.
- **RGB uint8 apa adanya**, sama dengan `build_patches.py` (tanpa stretch
  per tile). Input non-uint8 ditolak dengan pesan jelas.
- **Tanpa CuPy**: akumulasi di RAM CPU, GPU hanya untuk model. Kebutuhan RAM
  kanvas ≈ (NUM_CLASSES + 1) × lebar × tinggi × 4 byte, dicetak saat jalan.

Parameter default ada di `config.py` (`INFERENCE_TILE_SIZE`,
`INFERENCE_TILE_OVERLAP`, `INFERENCE_PATCH_OVERLAP`, `INFERENCE_BATCH_SIZE`)
dan bisa ditimpa lewat argumen. Overlap patch 75% sekitar 10× lebih berat
dari 25%; pakai `--patch-overlap 256` (50%) kalau terlalu lambat.

**Perbaikan artefak tile** (`inference/fix_tile_artifacts.py`): mendeteksi
blob kotak (axis-aligned) yang salah kelas secara utuh lewat
**rectangularity** (luas blob / luas bounding box, tanpa lubang) dan
**neighbor purity** (proporsi tetangga yang didominasi satu kelas lain), lalu
merelabelnya ke kelas tetangga. Sekarang **default mati**, karena margin
konteks + blending probabilitas sudah menangani penyebab utamanya. Aktifkan
dengan `--fix-tile-artifacts` (sensitivitas: `--tile-fix-rectangularity`,
`--tile-fix-purity`). Untuk SHP hasil inference yang **sudah ada**:

```bash
python inference/fix_tile_artifacts.py --input hasil_prediksi.shp --pixel-size 0.3
```

### 7. Uji akurasi lapangan (opsional)

```bash
python accuracy_assessment/grid_sampling_assessment.py   # dari nol, grid sampling otomatis
python accuracy_assessment/final_grid_assessment.py      # pakai grid final yang sudah ada
```

Kedua script ini **tidak** memakai `config.py` — edit langsung bagian
`KONFIGURASI` di bagian atas file (path SHP Ground Truth, SHP hasil
deteksi AI, dan untuk `final_grid_assessment.py` juga path SHP grid) sesuai
job/site yang sedang diuji.

`grid_sampling_assessment.py` menghitung ukuran sampel grid secara
statistik (rumus Cochran + Finite Population Correction), bukan angka
manual, lalu memilih grid representatif dengan stratifikasi zona spasial x
kepadatan tanaman.

## Konfigurasi (`config.py`)

Semua path folder, parameter CLAHE, ukuran patch, arsitektur model,
hyperparameter training, loss function, dan parameter inference ada di
satu class `Config`. Lihat komentar per-section di file tersebut. Method
berguna:

- `Config.diagnose()` — cek struktur folder `data/raw/` & kolom kelas SHP.
- `Config.summary()` — cetak ringkasan konfigurasi training saat ini.
- `Config.validate_config()` — validasi konsistensi config sebelum training.

## Catatan hasil refactor

- **Struktur folder**: sebelumnya semua script flat di root; sekarang
  dikelompokkan per tahap (`data_qc/`, `preprocessing/`, `training/`,
  `inference/`, `accuracy_assessment/`). `config.py` tetap di root karena
  dipakai hampir semua modul.
- **Bug diperbaiki**: `preprocessing.py` (kini `preprocessing/build_patches.py`)
  memanggil `config.RAW_DIR` yang **tidak ada** di `Config` (yang benar
  `RAW_DATA_DIR`) — akan crash `AttributeError` jika `run_preprocessing()`
  dipanggil langsung. Sudah diperbaiki.
- **Versi duplikat dihapus**: sebelumnya ada 3 versi script inference
  (`inference.py`, `inference_new.py`, `inference_old.py`) dan 2 versi
  script uji akurasi (`uji.py`, `uji1.py`) yang isinya saling tumpang
  tindih (evolusi dari waktu ke waktu). Atas pilihan pemilik project:
  - Inference: dipakai isi **`inference.py`** (kini `inference/run_inference.py`).
    `inference_new.py` (versi dengan normalisasi RGB stretch global,
    tile-halo, dan blending di level probabilitas, bukan argmax) dan
    `inference_old.py` **dihapus**, bukan diarsipkan.
  - Uji akurasi: dipakai isi **`uji1.py`** (kini
    `accuracy_assessment/grid_sampling_assessment.py`, dengan rumus
    Cochran+FPC untuk ukuran sampel otomatis). `uji.py` (versi manual
    `TARGET_N_GRID`) **dihapus**. `uji2.py` (kini `final_grid_assessment.py`)
    tetap dipertahankan karena alur kerjanya berbeda (pakai grid yang
    sudah final, bukan duplikat).
- **`check_tif_metadata.py`** (dulu `deb.py`) diubah dari script dengan
  path hardcoded jadi CLI (`--input`, `--target-resolution`) supaya bisa
  dipakai untuk file manapun.
- **`requirements.txt`** sebelumnya tersimpan dengan encoding UTF-16
  (hasil `pip freeze` di PowerShell) sehingga tidak valid dibaca `pip
  install -r`. Sudah ditulis ulang UTF-8, dikelompokkan per kategori,
  dan ditambah `openpyxl` (dependency nyata dari `pandas.ExcelWriter` di
  `accuracy_assessment/*.py` yang sebelumnya tidak tercatat).
