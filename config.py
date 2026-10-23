"""
config.py
=========
Konfigurasi terpusat untuk seluruh pipeline SegFormer 7-Band Land Cover.

Semua tahap (data_qc, preprocessing, training, inference, accuracy_assessment)
mengimpor `Config` dari modul ini, sehingga path folder, parameter model,
dan hyperparameter training hanya perlu diubah di SATU tempat.

File ini tetap berada di root project (bukan di dalam subfolder) supaya
`from config import Config` konsisten dipanggil dari script manapun,
termasuk yang berada di dalam subfolder (lihat bootstrap sys.path di
setiap script subfolder).
"""

import sys
from pathlib import Path
from rasterio.warp import Resampling

# Banyak script di pipeline ini mencetak emoji (✅ ❌ 🔍 dst) untuk log yang
# mudah dibaca. Di Windows, console default sering memakai codepage lama
# (cp1252/cp437) yang tidak bisa encode emoji -> UnicodeEncodeError saat
# print(). Paksa stdout/stderr ke UTF-8 di sini (dijalankan sekali saat
# config diimpor, yaitu di awal hampir semua script) supaya semua script
# yang mengimpor Config aman dari crash tersebut.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class Config:
    # ── Struktur folder project ─────────────────────────────────────
    BASE_DIR            = Path(__file__).parent
    DATA_DIR            = BASE_DIR / "data"
    RAW_DATA_DIR        = DATA_DIR / "raw"          # data/raw/<site>/*.tif + *.shp
    PROCESSED_DATA_DIR  = DATA_DIR / "processed"
    OUTPUT_DIR          = BASE_DIR / "outputs"

    CLAHE_OUTPUT_DIR    = PROCESSED_DATA_DIR / "clahe_enhanced"
    PATCHES_DIR         = PROCESSED_DATA_DIR / "patches"
    MASKS_DIR           = PROCESSED_DATA_DIR / "masks"

    MODEL_DIR           = OUTPUT_DIR / "models"
    LOG_DIR             = OUTPUT_DIR / "logs"
    CHECKPOINT_DIR      = OUTPUT_DIR / "checkpoints"

    # ── Parameter shapefile & CLAHE reference ───────────────────────
    CLAHE_REF_SITE      = None   # None = otomatis pilih site pertama yang punya TIF+SHP
    KELAS_COL           = "ID_NEW"   # nama kolom kelas di shapefile label

    TARGET_EPSG         = 32750
    TARGET_GSD          = 0.1
    RESAMPLING_METHOD   = Resampling.bilinear

    CLAHE_TILE_GRID     = (8, 8)
    CLAHE_CLIP_DEFAULT  = 2.0
    CLAHE_CLIP_BOUNDS   = (1.0, 15.0)
    CLAHE_CLIP_XTOL     = 0.2
    CLAHE_BLEND         = True
    CLAHE_SKIP_EXISTING = True

    TIFF_REPROJ_REF     = PROCESSED_DATA_DIR / "_ref_reproj.tif"

    # Pasangan kelas yang paling sering tertukar (dipakai clahe_enhancement.py
    # untuk optimasi clipLimit CLAHE via jarak Jeffries-Matusita/JM)
    CONFUSING_PAIRS = [
        (1, 2),
        (2, 6),
        (4, 5),
    ]

    IMAGE_EXT       = ".tif"
    SHAPEFILE_EXT   = ".shp"

    # ── Definisi band & kelas ────────────────────────────────────────
    BAND_NAMES      = ['R', 'G', 'B', 'ExG', 'VARI', 'Edge', 'Texture']
    NUM_CHANNELS    = 7

    NUM_CLASSES     = 10

    CLASS_COLORS = {
        0: [0, 0, 0],          # Background - Black
        1: [204, 0, 153],      # Bangunan - Magenta
        2: [255, 210, 0],      # Jalan - Bright Yellow
        3: [222, 205, 170],    # Lahan Terbuka - Cream
        4: [135, 206, 235],    # Badan Air - Light Sky Blue
        5: [0, 200, 100],      # Vegetasi Tinggi - Emerald Green
        6: [160, 230, 50],     # Vegetasi Rendah - Lime Green
        7: [0, 83, 58],        # Sawit - Bottle Green
        8: [153, 102, 255],    # Pinang - Violet
        9: [160, 65, 70],      # Lahan Terbakar - Muted Red-Grey
    }

    CLASS_NAMES = {
        0: 'Background',
        1: 'Bangunan',
        2: 'Jalan',
        3: 'Lahan Terbuka',
        4: 'Badan Air',
        5: 'Vegetasi Tinggi',
        6: 'Vegetasi Rendah',
        7: 'Sawit',
        8: 'Pinang',
        9: 'Lahan Terbakar'
    }    

    # ── Parameter patch extraction (preprocessing) ───────────────────
    PATCH_SIZE          = 512
    OVERLAP             = 128
    STRIDE              = PATCH_SIZE - OVERLAP

    SAVE_DTYPE_IMAGE    = "float16"
    SAVE_DTYPE_MASK     = "uint8"

    NORMALIZE           = True
    NORMALIZE_MEAN      = [0.485, 0.456, 0.406, 0.5, 0.5, 0.5, 0.5]
    NORMALIZE_STD       = [0.229, 0.224, 0.225, 0.25, 0.25, 0.25, 0.25]

    MIN_VALID_PIXELS    = 0.3
    IGNORE_INDEX        = 255

    # ── Arsitektur model ──────────────────────────────────────────────
    MODEL_NAME          = "nvidia/segformer-b2-finetuned-ade-512-512"
    ENCODER_NAME        = "mit_b2"
    PRETRAINED          = True
    FREEZE_ENCODER      = False
    DECODER_HIDDEN_SIZE = 256

    # ── Hyperparameter training ───────────────────────────────────────
    # Disetel untuk Radeon RX 9070 16 GB (versi AMD/ROCm).
    # Versi CUDA lama memakai batch 2 untuk RTX 3050 6 GB. SegFormer-B2 512x512
    # + AMP ~1.5 GB per sampel -> batch 8 ~ 12 GB (aman di 16 GB). Kalau muncul
    # "HIP out of memory", turunkan ke 6 atau 4.
    # Catatan: batch lebih besar = update gradient lebih jarang per epoch.
    # Kalau ingin perilaku persis seperti training lama, set BATCH_SIZE = 2.
    BATCH_SIZE          = 8
    NUM_EPOCHS          = 150
    LEARNING_RATE       = 6e-5
    WEIGHT_DECAY        = 0.01

    OPTIMIZER           = "adamw"
    BETAS               = (0.9, 0.999)
    EPS                 = 1e-8

    LR_SCHEDULER        = "cosine"
    LR_POWER            = 0.9
    WARMUP_EPOCHS       = 5
    WARMUP_LR           = 1e-6

    # ── Loss function ──────────────────────────────────────────────────
    LOSS_FUNCTION       = "focal_dice"

    FOCAL_ALPHA         = 0.25
    FOCAL_GAMMA         = 3.0

    DICE_SMOOTH         = 1.0

    FOCAL_WEIGHT        = 0.5
    DICE_WEIGHT         = 0.5

    # Class weight per kelas untuk mengatasi imbalance (index = class id).
    # Dipakai oleh loss "crossentropy", "focal", dan bagian focal dari
    # "focal_dice" (Dice sudah seimbang per kelas, jadi tidak diberi bobot).
    #
    #   "auto" -> dihitung OTOMATIS dari mask split TRAIN setiap kali training
    #             dijalankan (data_qc/calculate_class_weights.py). Hasilnya
    #             di-cache di CLASS_WEIGHT_CACHE dan hanya dihitung ulang kalau
    #             dataset berubah — tidak perlu copy-paste lagi.
    #   None   -> tanpa class weights
    #   [w0, w1, ...] -> override manual (panjang harus = NUM_CLASSES)
    CLASS_WEIGHTS       = "auto"

    # Metode hitung untuk mode "auto":
    #   "balanced" (inverse frequency), "sqrt" (inverse sqrt frequency, lebih
    #   lembut untuk imbalance ekstrem), "effective", "sklearn"
    CLASS_WEIGHT_METHOD = "balanced"
    CLASS_WEIGHT_CACHE  = LOG_DIR / "class_weights.json"

    # ── Augmentasi data ───────────────────────────────────────────────
    USE_AUGMENTATION        = True
    AUG_HORIZONTAL_FLIP     = 0.5
    AUG_VERTICAL_FLIP       = 0.5
    AUG_ROTATION            = 0.5
    AUG_ROTATION_LIMIT      = 15
    AUG_BRIGHTNESS          = 0.2
    AUG_CONTRAST            = 0.2

    # ── Split dataset & validasi ──────────────────────────────────────
    TRAIN_SPLIT             = 0.75
    VAL_SPLIT               = 0.25
    RANDOM_SEED             = 42
    VAL_FREQUENCY           = 1
    SAVE_BEST_ONLY          = False

    # Metrik untuk memilih best_model.pth & early stopping:
    #   "miou"  -> mean IoU per kelas (setiap kelas dihitung sama penting;
    #              lebih menjaga kelas kecil seperti Badan Air / Pemukiman)
    #   "fwiou" -> frequency-weighted IoU (IoU tiap kelas dibobot proporsi
    #              pixel kelas tsb di ground truth)
    # Keduanya (plus pixel accuracy) selalu dihitung & dilaporkan.
    BEST_MODEL_METRIC       = "miou"
    EARLY_STOPPING_PATIENCE = 10

    # ── Logging & checkpoint ───────────────────────────────────────────
    LOG_FREQUENCY   = 10
    TENSORBOARD     = False
    WANDB           = False
    WANDB_PROJECT   = "segformer-7band-landcover"
    SAVE_FREQUENCY  = 5
    MAX_CHECKPOINTS = 5

    # ── Parameter inference ────────────────────────────────────────────
    TEST_TIME_AUGMENTATION  = False
    INFERENCE_BATCH_SIZE    = 16    # patch per batch saat prediksi (RX 9070 16 GB; versi lama 4)
    INFERENCE_TILE_SIZE     = 2048   # ukuran tile baca raster (px)
    INFERENCE_TILE_OVERLAP  = 128    # overlap antar tile; separuhnya jadi margin konteks
    INFERENCE_PATCH_OVERLAP = 384    # overlap patch 512 px (75%); 256 = 50% lebih cepat
    INFERENCE_SIGMA_RATIO   = 6.0    # ketajaman bobot Gaussian blending patch

    # ── Hardware ────────────────────────────────────────────────────────
    # PyTorch ROCm (AMD) tetap memakai nama device "cuda" (API torch.cuda
    # dipetakan ke HIP), jadi "cuda" di sini = GPU AMD. "cpu" untuk tanpa GPU.
    DEVICE          = "cuda"
    # None -> otomatis: iGPU Radeon bawaan Ryzen 7 7700 (gfx1036, cuda:0)
    # dilewati, lalu pilih GPU diskrit berarsitektur GPU_ARCH. Isi angka
    # (0, 1, ...) untuk memaksa index tertentu.
    GPU_INDEX       = None
    # Arsitektur GPU target = wheel torch[device-gfx1201] (RX 9070). Kernel
    # hanya ada untuk arsitektur ini. None -> GPU diskrit dengan VRAM terbesar.
    GPU_ARCH        = "gfx1201"
    # MIOpen (cuDNN-nya AMD). False = kernel native PyTorch. Wheel ROCm 10.1
    # Windows gagal mengompilasi kernel BatchNorm MIOpen ("'type_traits' file
    # not found" -> miopenStatusUnknownError), dan native malah lebih cepat
    # di RX 9070 (inference batch 16: 199 vs 372 ms, training batch 8: 274 vs 461 ms).
    USE_MIOPEN      = False
    # Windows: tiap worker memuat ulang torch + DLL ROCm (~1-2 GB commit RAM).
    # RAM 32 GB -> 4 worker masih aman. Turunkan ke 2 kalau muncul
    # "WinError 1455 paging file too small".
    NUM_WORKERS     = 4
    PIN_MEMORY      = True
    MIXED_PRECISION = True
    # "bfloat16" (default, native di RDNA4, tanpa GradScaler, lebih stabil)
    # atau "float16" (perilaku sama dengan versi CUDA lama, pakai GradScaler).
    AMP_DTYPE       = "bfloat16"
    MULTI_GPU       = False
    GPU_IDS         = [0]

    GRADIENT_ACCUMULATION_STEPS = 1

    @classmethod
    def make_dirs(cls):
        for d in [
            cls.PROCESSED_DATA_DIR,
            cls.CLAHE_OUTPUT_DIR,
            cls.PATCHES_DIR,
            cls.MASKS_DIR,
            cls.MODEL_DIR,
            cls.LOG_DIR,
            cls.CHECKPOINT_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)


    @classmethod
    def diagnose(cls):
        """Cetak info struktur folder & file yang ditemukan."""
        print("\n" + "="*60)
        print("🔍 CONFIG DIAGNOSIS")
        print("="*60)
        print(f"  BASE_DIR       : {cls.BASE_DIR}")
        print(f"  RAW_DATA_DIR   : {cls.RAW_DATA_DIR}")
        print(f"  CLAHE_REF_SITE : {cls.CLAHE_REF_SITE}")
        print(f"  KELAS_COL      : {cls.KELAS_COL}")
        print()

        if not cls.RAW_DATA_DIR.exists():
            print(f"  ❌ RAW_DATA_DIR tidak ditemukan: {cls.RAW_DATA_DIR}")
            print(f"     → Buat folder: {cls.RAW_DATA_DIR}")
            return

        sites = sorted([d for d in cls.RAW_DATA_DIR.iterdir() if d.is_dir()])
        if not sites:
            print(f"  ❌ Tidak ada sub-folder site di {cls.RAW_DATA_DIR}")
            print(f"     → Struktur yang diharapkan:")
            print(f"        data/raw/")
            print(f"        └── nama_site/")
            print(f"            ├── gambar.tif")
            print(f"            └── label.shp")
            return

        print(f"  ✅ Ditemukan {len(sites)} site:")
        for site in sites:
            tifs = list(site.glob("*.tif"))
            shps = list(site.glob("*.shp"))
            tif_ok = "✅" if tifs else "❌"
            shp_ok = "✅" if shps else "❌"
            print(f"     📁 {site.name}")
            print(f"        {tif_ok} TIF : {len(tifs)} file(s)")
            print(f"        {shp_ok} SHP : {len(shps)} file(s)")

            if shps:
                try:
                    import geopandas as gpd
                    gdf = gpd.read_file(shps[0])
                    cols = list(gdf.columns)
                    kelas_ok = "✅" if cls.KELAS_COL in cols else "❌"
                    print(f"        {kelas_ok} Kolom '{cls.KELAS_COL}' "
                          f"{'ditemukan' if cls.KELAS_COL in cols else 'TIDAK ADA'}")
                    if cls.KELAS_COL not in cols:
                        print(f"           Kolom tersedia: {cols}")
                except Exception as e:
                    print(f"        ⚠️  Gagal baca SHP: {str(e)[:50]}")

        print()
        if cls.CLAHE_REF_SITE is None:
            print(f"  💡 CLAHE_REF_SITE = None → akan pakai '{sites[0].name}' sebagai referensi")
        else:
            ref_dir = cls.RAW_DATA_DIR / cls.CLAHE_REF_SITE
            if ref_dir.exists():
                print(f"  ✅ CLAHE_REF_SITE '{cls.CLAHE_REF_SITE}' ditemukan")
            else:
                print(f"  ❌ CLAHE_REF_SITE '{cls.CLAHE_REF_SITE}' TIDAK ditemukan!")
                print(f"     → Set CLAHE_REF_SITE = None untuk otomatis,")
                print(f"       atau ganti ke salah satu: {[s.name for s in sites]}")
        print("="*60 + "\n")

    @classmethod
    def summary(cls):
        """Print ringkasan konfigurasi training."""
        print("\n" + "="*60)
        print("⚙️  TRAINING CONFIGURATION SUMMARY")
        print("="*60)
        print(f"  Model Architecture : {cls.MODEL_NAME}")
        print(f"  Input Channels     : {cls.NUM_CHANNELS} ({', '.join(cls.BAND_NAMES)})")
        print(f"  Number of Classes  : {cls.NUM_CLASSES}")
        print(f"  Patch Size         : {cls.PATCH_SIZE}x{cls.PATCH_SIZE}")
        print(f"  Overlap            : {cls.OVERLAP}")
        print()
        print("  Training Parameters:")
        print(f"    Batch Size       : {cls.BATCH_SIZE}")
        print(f"    Epochs           : {cls.NUM_EPOCHS}")
        print(f"    Learning Rate    : {cls.LEARNING_RATE}")
        print(f"    Optimizer        : {cls.OPTIMIZER.upper()}")
        print(f"    Weight Decay     : {cls.WEIGHT_DECAY}")
        print(f"    LR Scheduler     : {cls.LR_SCHEDULER}")
        print(f"    Warmup Epochs    : {cls.WARMUP_EPOCHS}")
        print()
        print("  Loss Function:")
        print(f"    Type             : {cls.LOSS_FUNCTION.upper()}")
        if cls.LOSS_FUNCTION == "focal_dice":
            print(f"    Focal Weight     : {cls.FOCAL_WEIGHT}")
            print(f"    Dice Weight      : {cls.DICE_WEIGHT}")
            print(f"    Focal Alpha      : {cls.FOCAL_ALPHA}")
            print(f"    Focal Gamma      : {cls.FOCAL_GAMMA}")
            print(f"    Dice Smooth      : {cls.DICE_SMOOTH}")
        cw = cls.CLASS_WEIGHTS
        cw_desc = (f'auto ({cls.CLASS_WEIGHT_METHOD})' if isinstance(cw, str)
                   else 'None' if cw is None else 'manual list')
        print(f"    Class Weights    : {cw_desc}")
        print()
        print("  Hardware:")
        print(f"    Device           : {cls.DEVICE}")
        gpu_auto = f"auto (diskrit, {cls.GPU_ARCH or 'VRAM terbesar'})"
        print(f"    GPU Index        : {cls.GPU_INDEX if cls.GPU_INDEX is not None else gpu_auto}")
        print(f"    MIOpen           : {'ON' if cls.USE_MIOPEN else 'OFF (kernel native PyTorch)'}")
        print(f"    Mixed Precision  : {cls.MIXED_PRECISION} ({cls.AMP_DTYPE})")
        print(f"    Num Workers      : {cls.NUM_WORKERS}")
        print(f"    Pin Memory       : {cls.PIN_MEMORY}")
        print()
        print("  Augmentation:")
        print(f"    Enabled          : {cls.USE_AUGMENTATION}")
        if cls.USE_AUGMENTATION:
            print(f"    H-Flip (p)       : {cls.AUG_HORIZONTAL_FLIP}")
            print(f"    V-Flip (p)       : {cls.AUG_VERTICAL_FLIP}")
            print(f"    Rotation (p)     : {cls.AUG_ROTATION} (±{cls.AUG_ROTATION_LIMIT}°)")
            print(f"    Brightness       : {cls.AUG_BRIGHTNESS}")
            print(f"    Contrast         : {cls.AUG_CONTRAST}")
        print()
        print("  Validation:")
        print(f"    Val Split        : {cls.VAL_SPLIT}")
        print(f"    Val Frequency    : Every {cls.VAL_FREQUENCY} epoch(s)")
        print(f"    Early Stopping   : {cls.EARLY_STOPPING_PATIENCE} epochs")
        print(f"    Best Model By    : {cls.BEST_MODEL_METRIC}")
        print()
        print("  Checkpointing:")
        print(f"    Save Frequency   : Every {cls.SAVE_FREQUENCY} epoch(s)")
        print(f"    Max Checkpoints  : {cls.MAX_CHECKPOINTS}")
        print(f"    TensorBoard      : {cls.TENSORBOARD}")
        print("="*60 + "\n")

    @classmethod
    def validate_config(cls):
        """Validasi konfigurasi sebelum training."""
        errors = []
        warnings = []

        # Validasi loss weights
        if cls.LOSS_FUNCTION == "focal_dice":
            total_weight = cls.FOCAL_WEIGHT + cls.DICE_WEIGHT
            if abs(total_weight - 1.0) > 0.01:
                warnings.append(
                    f"FOCAL_WEIGHT + DICE_WEIGHT = {total_weight:.2f} (disarankan = 1.0)"
                )

        # Validasi batch size
        if cls.BATCH_SIZE < 1:
            errors.append("BATCH_SIZE harus >= 1")

        # Validasi num_classes
        if cls.NUM_CLASSES != len(cls.CLASS_COLORS):
            errors.append(
                f"NUM_CLASSES ({cls.NUM_CLASSES}) != len(CLASS_COLORS) ({len(cls.CLASS_COLORS)})"
            )

        # Validasi class weights
        cw = cls.CLASS_WEIGHTS
        if isinstance(cw, str):
            if cw.lower() != "auto":
                errors.append(f'CLASS_WEIGHTS = "{cw}" tidak dikenal (pakai "auto", None, atau list)')
            elif cls.CLASS_WEIGHT_METHOD not in ("balanced", "sqrt", "effective", "sklearn"):
                errors.append(f"CLASS_WEIGHT_METHOD = {cls.CLASS_WEIGHT_METHOD!r} tidak dikenal")
        elif cw is not None and len(cw) != cls.NUM_CLASSES:
            errors.append(f"len(CLASS_WEIGHTS) ({len(cw)}) != NUM_CLASSES ({cls.NUM_CLASSES})")
        if cls.CLASS_WEIGHTS is not None and cls.LOSS_FUNCTION == "dice":
            warnings.append('LOSS_FUNCTION = "dice" tidak memakai CLASS_WEIGHTS')

        if cls.BEST_MODEL_METRIC not in ("miou", "fwiou"):
            errors.append(f'BEST_MODEL_METRIC harus "miou" atau "fwiou", bukan {cls.BEST_MODEL_METRIC!r}')

        if str(cls.AMP_DTYPE).lower() not in ("bfloat16", "bf16", "float16", "fp16"):
            errors.append(f'AMP_DTYPE harus "bfloat16" atau "float16", bukan {cls.AMP_DTYPE!r}')

        # Validasi paths
        if not cls.RAW_DATA_DIR.exists():
            warnings.append(f"RAW_DATA_DIR tidak ditemukan: {cls.RAW_DATA_DIR}")

        # Print hasil validasi
        if errors:
            print("\n" + "="*60)
            print("❌ CONFIGURATION ERRORS:")
            for err in errors:
                print(f"  • {err}")
            print("="*60 + "\n")
            raise ValueError("Konfigurasi tidak valid! Perbaiki error di atas.")

        if warnings:
            print("\n" + "="*60)
            print("⚠️  CONFIGURATION WARNINGS:")
            for warn in warnings:
                print(f"  • {warn}")
            print("="*60 + "\n")

        return True
