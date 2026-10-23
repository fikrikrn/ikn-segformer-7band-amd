@echo off
REM ==================================================================
REM setup_venv.bat - versi AMD / ROCm
REM Buat virtual environment Python 3.12 (venv) + PyTorch ROCm untuk
REM AMD Radeon RX 9070 (gfx1201), lalu install dependency pipeline.
REM
REM Jalankan dari root project (CMD, bukan PowerShell):   setup_venv.bat
REM
REM Prasyarat:
REM   1. Python 3.12 64-bit  (cek:  py -3.12 --version)
REM      https://www.python.org/downloads/  -> centang "Add to PATH"
REM   2. Driver AMD Software: Adrenalin Edition terbaru (minimal 26.2.2)
REM   3. Hapus instalasi lama "AMD HIP SDK" kalau ada, lalu restart
REM      (DLL HIP lama di PATH sering bentrok dengan wheel ROCm).
REM ==================================================================
setlocal
cd /d "%~dp0"

py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.12 tidak ditemukan. Install dari https://www.python.org/downloads/
    echo         lalu jalankan ulang script ini.
    exit /b 1
)

if exist venv (
    echo [INFO] venv sudah ada, dipakai ulang. Hapus folder venv kalau mau buat dari nol.
) else (
    echo [1/5] Membuat venv dengan Python 3.12 ...
    py -3.12 -m venv venv || exit /b 1
)

call venv\Scripts\activate.bat || exit /b 1

echo [2/5] Upgrade pip ...
python -m pip install --upgrade pip setuptools wheel || exit /b 1

REM Buang torch lama (CPU/CUDA) kalau venv dipakai ulang
python -m pip uninstall -y torch torchvision torchaudio >nul 2>&1

echo [3A/5] Install PyTorch ROCm untuk gfx1201 (RX 9070) ...
REM torch & torchvision WAJIB dikunci berpasangan: metadata torchvision ROCm tidak
REM mengunci versi torch, jadi tanpa pin pip ambil torch terbaru (mis. 2.14) yang
REM belum punya torchvision pasangannya -> "operator torchvision::nms does not exist".
REM Pasangan resmi: torch 2.13 <-> torchvision 0.28, torch 2.14 <-> torchvision 0.29.
python -m pip install --no-cache-dir --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1201]==2.13.0+rocm10.1.0" "torchvision[device-gfx1201]==0.28.0+rocm10.1.0"
if not errorlevel 1 goto torch_ok

echo.
echo [3B/5] Index ROCm terbaru gagal -> fallback ke rilis resmi ROCm 7.2.1 (torch 2.9.1) ...
set "R=https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"
python -m pip install --no-cache-dir ^
    %R%/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl ^
    %R%/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl ^
    %R%/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl ^
    %R%/rocm-7.2.1.tar.gz || exit /b 1
python -m pip install --no-cache-dir ^
    %R%/torch-2.9.1%%2Brocm7.2.1-cp312-cp312-win_amd64.whl ^
    %R%/torchvision-0.24.1%%2Brocm7.2.1-cp312-cp312-win_amd64.whl || exit /b 1

:torch_ok
echo [4/5] Install dependency lain dari requirements.txt (torch ROCm dikunci) ...
python -c "import importlib.metadata as m; open('constraints_torch.txt','w').write(''.join(f'{p}=={m.version(p)}\n' for p in ('torch','torchvision')))" || exit /b 1
type constraints_torch.txt
python -m pip install -r requirements.txt -c constraints_torch.txt || exit /b 1

echo.
echo [5/5] ===== Cek instalasi GPU =====
python gpu_utils.py
if errorlevel 1 (
    echo.
    echo [WARN] GPU AMD belum terdeteksi oleh PyTorch. Cek:
    echo        - driver Adrenalin sudah terbaru, lalu restart
    echo        - tidak ada AMD HIP SDK lama di PATH
    echo        - Smart App Control Windows tidak memblokir DLL ROCm
)
echo.
echo Selesai. Aktifkan venv dengan:  venv\Scripts\activate
endlocal
