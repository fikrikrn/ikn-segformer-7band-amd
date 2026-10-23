@echo off
REM ==================================================================
REM copy_data_from_cuda.bat
REM Salin dataset & model dari project versi CUDA (..\vale-segformer-7band)
REM ke folder versi AMD ini, supaya bisa langsung training/inference.
REM
REM Yang disalin (~2.5 GB):
REM   data\                          raw + processed (patches, masks, clahe)
REM   outputs\models\                best_model.pth
REM   outputs\logs\                  class_weights.json, history, metrics
REM   outputs\checkpoints\           (hanya kalau mau resume training)
REM Tidak disalin: venv, .git, dev, outputs\job_* (hasil inference lama).
REM Folder sumber TIDAK diubah / dihapus (robocopy tanpa /MOVE /MIR).
REM ==================================================================
setlocal
cd /d "%~dp0"
set "SRC=%~dp0..\vale-segformer-7band"

if not exist "%SRC%\data" (
    echo [ERROR] Folder sumber tidak ditemukan: %SRC%
    exit /b 1
)

echo [1/3] data\ ...
robocopy "%SRC%\data" "data" /E /NFL /NDL /NJH /R:1 /W:1
echo [2/3] outputs\models + outputs\logs ...
robocopy "%SRC%\outputs\models" "outputs\models" /E /NFL /NDL /NJH /R:1 /W:1
robocopy "%SRC%\outputs\logs" "outputs\logs" /E /NFL /NDL /NJH /R:1 /W:1

choice /M "[3/3] Salin juga outputs\checkpoints (untuk resume training, ~1.6 GB)"
if errorlevel 2 goto done
robocopy "%SRC%\outputs\checkpoints" "outputs\checkpoints" /E /NFL /NDL /NJH /R:1 /W:1

:done
echo Selesai.
endlocal
exit /b 0
