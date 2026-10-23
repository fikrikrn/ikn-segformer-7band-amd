# data/processed/patches/

**Output otomatis** dari `preprocessing/build_patches.py` — jangan diisi manual.

Berisi `<site>/*.npy`, patch citra 7-band (`R, G, B, ExG, VARI, Edge,
Texture`) ukuran `Config.PATCH_SIZE`, siap dipakai oleh
`training/train_model.py`. Berpasangan 1:1 dengan file di
`data/processed/masks/` (nama file sama).

Isi folder ini tidak ikut di-commit ke git (lihat `.gitignore`).
