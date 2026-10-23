# outputs/checkpoints/

**Output otomatis** dari `training/train_model.py` — jangan diisi manual.

Berisi `checkpoint_epoch_<n>.pth` periodik (setiap `Config.SAVE_FREQUENCY`
epoch), dipakai untuk resume training. Jumlah checkpoint tersimpan dibatasi
`Config.MAX_CHECKPOINTS` (yang paling lama otomatis dihapus).

Isi folder ini tidak ikut di-commit ke git (lihat `.gitignore`).
