# data/processed/masks/

**Output otomatis** dari `preprocessing/build_patches.py` — jangan diisi manual.

Berisi `<site>/*.npy`, mask kelas (hasil rasterisasi SHP) yang sudah
dipotong jadi patch, berpasangan 1:1 dengan file di
`data/processed/patches/` (nama file sama). Dipakai oleh
`training/train_model.py` dan script `data_qc/*.py`.

Isi folder ini tidak ikut di-commit ke git (lihat `.gitignore`).
