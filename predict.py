"""
predict.py
==========
Pintasan ke inference/run_inference.py (isi asli predict.py sudah dilebur ke
sana dan disesuaikan dengan struktur project ini). Semua argumen sama:

    python predict.py --input citra.tif --resolution 0.1
    python predict.py --input folder/ --model outputs/models/best_model.pth
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from inference.run_inference import main

if __name__ == "__main__":
    main()
