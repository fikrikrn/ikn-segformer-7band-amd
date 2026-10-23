# data/raw/

Taruh data mentah di sini, **satu subfolder per site**:

```
data/raw/
└── nama_site/
    ├── nama_site.tif   # citra RGB (GeoTIFF)
    └── nama_site.shp   # label kelas (+ .dbf, .shx, .prj, dst)
```

- Nama file `.tif` dan `.shp` **harus sama** dengan nama folder site-nya
  (dipakai oleh `preprocessing/build_patches.py` untuk mencari pasangannya).
- Kolom kelas di `.shp` diatur lewat `Config.KELAS_COL` (default `"ID_NEW"`, berisi ID kelas 0–9 sesuai `Config.CLASS_NAMES`).
- Cek struktur folder ini kapan saja dengan:
  ```bash
  python -c "from config import Config; Config.diagnose()"
  ```

Isi folder ini tidak ikut di-commit ke git (lihat `.gitignore`) — file ini
hanya penanda supaya foldernya ada di repo.
