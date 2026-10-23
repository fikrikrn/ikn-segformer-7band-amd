"""
gpu_utils.py
============
Helper GPU terpusat untuk versi AMD (ROCm) pipeline SegFormer 7-Band.

Kenapa perlu file ini?
  PyTorch build ROCm memakai HIP, tapi API-nya TETAP bernama `torch.cuda`
  dan device-nya tetap `"cuda"`. Jadi sebagian besar kode lama sudah jalan,
  hanya beberapa hal yang perlu disesuaikan:

  1. Pemilihan GPU. Ryzen 7 7700 punya iGPU Radeon (gfx1036) yang ikut
     terdeteksi sebagai `cuda:0`, bukan RX 9070. Di sini iGPU dilewati dan
     GPU dipilih berdasarkan arsitektur Config.GPU_ARCH ("gfx1201" = RX 9070),
     atau dipaksa lewat Config.GPU_INDEX.
  2. Mixed precision. RDNA4 (RX 9070, gfx1201) mendukung bfloat16 secara
     native. bf16 tidak butuh GradScaler dan lebih stabil dari fp16.
     Config.AMP_DTYPE = "bfloat16" | "float16".
  3. Pesan error/diagnostik menyebut ROCm/HIP, bukan CUDA (torch.version.cuda
     bernilai None di build ROCm; versinya ada di torch.version.hip).
  4. MIOpen (pengganti cuDNN di AMD) dimatikan secara default
     (Config.USE_MIOPEN = False): kernel BatchNorm-nya gagal dikompilasi di
     wheel ROCm Windows, dan kernel native PyTorch lebih cepat untuk SegFormer.
     Variabel environment MIOpen & alokator tetap diset default yang aman
     untuk Windows sebelum kernel pertama jalan.

Pemakaian:
    from gpu_utils import setup_gpu_env, get_device, amp_autocast, make_grad_scaler
    setup_gpu_env()                 # panggil sedini mungkin
    device = get_device(Config)     # torch.device("cuda:N") / "cpu"
    with amp_autocast(device, Config):
        logits = model(x)

Cek cepat instalasi:  python gpu_utils.py
"""

import os
import contextlib


def setup_gpu_env():
    """Set default env var ROCm/MIOpen. Tidak menimpa nilai yang sudah diset
    user. Aman dipanggil berkali-kali. Idealnya sebelum operasi GPU pertama."""
    # MIOpen "find" mode FAST: hindari tuning kernel konvolusi yang lama sekali
    # di run pertama (default NORMAL bisa membuat epoch pertama sangat lambat).
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
    # Kurangi log MIOpen yang berisik di console
    os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")
    # Alokator memori: expandable_segments mengurangi fragmentasi VRAM, TAPI
    # jangan di Windows. Di ROCm Windows tiap segmen me-reserve ~1.1x VRAM
    # (~18 GB) yang ikut di-commit ke RAM sistem -> RAM proses ~27 GB (commit
    # ~38 GB) sebelum model jalan, dan model.to(GPU) jadi ~35 detik.
    if os.name != "nt":
        os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")


def backend_name():
    import torch
    if getattr(torch.version, "hip", None):
        return f"ROCm/HIP {torch.version.hip}"
    if getattr(torch.version, "cuda", None):
        return f"CUDA {torch.version.cuda}"
    return "CPU-only build"


def is_rocm():
    import torch
    return bool(getattr(torch.version, "hip", None))


def _arch(props):
    """'gfx1201:sramecc-:xnack-' -> 'gfx1201'."""
    return getattr(props, "gcnArchName", "").split(":")[0]


def _pick_gpu_index(config=None):
    """Index GPU yang dipakai:
      1. Config.GPU_INDEX kalau diisi (paksa manual).
      2. Kalau tidak: iGPU dilewati, lalu dari GPU diskrit dipilih yang
         arsitekturnya = Config.GPU_ARCH ("gfx1201" = RX 9070, sesuai wheel
         torch[device-gfx1201]). GPU_ARCH = None -> GPU diskrit VRAM terbesar.
    Sengaja tidak murni "VRAM terbesar": iGPU memakai RAM sistem sebagai
    memorinya, jadi total_memory-nya ikut naik kalau RAM di-upgrade dan bisa
    mengalahkan RX 9070."""
    import torch
    n = torch.cuda.device_count()
    forced = getattr(config, "GPU_INDEX", None) if config is not None else None
    if forced is not None:
        if not (0 <= int(forced) < n):
            raise RuntimeError(f"Config.GPU_INDEX = {forced} tapi hanya ada {n} GPU terdeteksi.")
        return int(forced)

    props = [torch.cuda.get_device_properties(i) for i in range(n)]
    want_arch = getattr(config, "GPU_ARCH", None) if config is not None else None
    candidates = [i for i, p in enumerate(props)
                  if not getattr(p, "is_integrated", 0)
                  and (want_arch is None or _arch(p) == want_arch)]
    if not candidates:
        found = ", ".join(
            f"[{i}] {p.name} {_arch(p)}{' (iGPU)' if getattr(p, 'is_integrated', 0) else ''}"
            for i, p in enumerate(props))
        target = f"GPU diskrit {want_arch}" if want_arch else "GPU diskrit"
        raise RuntimeError(
            f"Tidak ada {target} yang terdeteksi. GPU yang ada: {found or '-'}.\n"
            "Set Config.GPU_ARCH = None (GPU diskrit mana pun) atau "
            "Config.GPU_INDEX = <index> untuk memaksa.")
    return max(candidates, key=lambda i: props[i].total_memory)


def get_device(config=None, strict=False):
    """Kembalikan torch.device sesuai Config.DEVICE.

    Config.DEVICE = "cuda" (= GPU AMD lewat ROCm) atau "cpu".
    strict=True -> raise error kalau GPU diminta tapi tidak ada (dipakai
    training, supaya tidak diam-diam jalan di CPU)."""
    import torch
    setup_gpu_env()
    want_gpu = str(getattr(config, "DEVICE", "cuda")).startswith("cuda")
    if not want_gpu:
        return torch.device("cpu")

    if not torch.cuda.is_available():
        msg = (
            f"Config.DEVICE = 'cuda' tapi GPU AMD tidak terdeteksi "
            f"(torch {torch.__version__}, backend: {backend_name()}).\n"
            "Penyebab umum:\n"
            "  - PyTorch yang terinstal versi CPU/CUDA, bukan ROCm -> jalankan ulang setup_venv.bat\n"
            "  - Driver AMD Adrenalin belum versi yang didukung ROCm on Windows\n"
            "  - Python bukan 3.12 (wheel ROCm Windows hanya cp312)\n"
            "Cek detail dengan:  python gpu_utils.py\n"
            "Atau set DEVICE = 'cpu' di config.py kalau memang ingin jalan di CPU."
        )
        if strict:
            raise RuntimeError(msg)
        print("⚠️  " + msg.splitlines()[0] + " -> jalan di CPU (lambat).")
        return torch.device("cpu")

    idx = _pick_gpu_index(config)
    torch.cuda.set_device(idx)
    if is_rocm():
        # Di build ROCm, flag cudnn = MIOpen. Default MATI (Config.USE_MIOPEN):
        # wheel ROCm Windows gagal mengompilasi kernel BatchNorm MIOpen
        # (hiprtc: 'type_traits' file not found -> miopenStatusUnknownError),
        # dan kernel native PyTorch malah lebih cepat untuk SegFormer.
        torch.backends.cudnn.enabled = bool(getattr(config, "USE_MIOPEN", False))
    return torch.device(f"cuda:{idx}")


def device_label(device):
    import torch
    if device.type != "cuda":
        return "CPU"
    idx = device.index if device.index is not None else torch.cuda.current_device()
    p = torch.cuda.get_device_properties(idx)
    arch = getattr(p, "gcnArchName", "")
    arch = f", {arch.split(':')[0]}" if arch else ""
    return f"{p.name} [{p.total_memory / 1024**3:.1f} GB{arch}] | {backend_name()}"


def amp_dtype(device, config=None):
    """dtype autocast. None kalau AMP mati / device CPU."""
    import torch
    if device.type != "cuda" or not bool(getattr(config, "MIXED_PRECISION", True)):
        return None
    name = str(getattr(config, "AMP_DTYPE", "bfloat16")).lower()
    if name in ("bf16", "bfloat16"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print("⚠️  GPU tidak mendukung bfloat16 -> AMP pakai float16")
    return torch.float16


def amp_autocast(device, config=None):
    """Context manager autocast (atau no-op kalau AMP tidak dipakai)."""
    import torch
    dt = amp_dtype(device, config)
    if dt is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dt)


def make_grad_scaler(device, config=None):
    """GradScaler hanya untuk float16. bfloat16 punya range eksponen sama
    dengan float32 sehingga tidak perlu loss scaling -> return None."""
    import torch
    if amp_dtype(device, config) is not torch.float16:
        return None
    return torch.amp.GradScaler("cuda")


def _selftest():
    import sys
    import time
    import torch
    print(f"Python  : {sys.version.split()[0]}")
    print(f"Torch   : {torch.__version__}  ({backend_name()})")
    print(f"GPU ok  : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return 1
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name}  {p.total_memory / 1024**3:.1f} GB  {getattr(p, 'gcnArchName', '')}")

    class _C:
        DEVICE = "cuda"
        GPU_ARCH = "gfx1201"
        MIXED_PRECISION = True
        AMP_DTYPE = "bfloat16"
    dev = get_device(_C, strict=True)
    print(f"Dipakai : {device_label(dev)}")
    print(f"bf16    : {torch.cuda.is_bf16_supported()}")

    # Uji kecil: conv + matmul + backward dengan AMP (memastikan MIOpen & hipBLAS jalan)
    x = torch.randn(2, 7, 512, 512, device=dev)
    conv = torch.nn.Conv2d(7, 64, 7, stride=4, padding=3).to(dev)
    t0 = time.time()
    with amp_autocast(dev, _C):
        y = conv(x).flatten(2).transpose(1, 2)
        z = (y @ y.transpose(1, 2)).mean()
    z.float().backward()
    torch.cuda.synchronize()
    print(f"Uji conv/matmul/backward OK ({time.time() - t0:.2f} s, termasuk warm-up)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
