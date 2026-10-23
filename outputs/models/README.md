# outputs/models/

**Output otomatis** dari `training/train_model.py` — jangan diisi manual.

Berisi `best_model.pth`, checkpoint dengan val IoU terbaik selama training.
Ini yang dipakai sebagai default `--model` oleh `inference/run_inference.py` (dan `predict.py`).

Isi folder ini tidak ikut di-commit ke git (lihat `.gitignore`).
