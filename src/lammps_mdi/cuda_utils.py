"""
cuda_utils.py — CUDA version detection and PyTorch installation guidance.

Used by the `lammps-mdi check` and `lammps-mdi install-torch` commands,
and by the installation documentation generator.
"""

import re
import subprocess
import sys
from typing import Optional


# Map (cuda_major, cuda_minor_minimum) -> PyTorch wheel tag, newest first.
# Update this table when new PyTorch CUDA builds are released.
_CUDA_TO_TORCH_TAG = [
    ((12, 8), "cu128"),
    ((12, 6), "cu126"),
    ((12, 4), "cu124"),
    ((12, 1), "cu121"),
    ((11, 8), "cu118"),
]

TORCH_INDEX_BASE = "https://download.pytorch.org/whl"


def detect_cuda_version() -> tuple[Optional[int], Optional[int]]:
    """Return the (major, minor) CUDA version from nvidia-smi, or (None, None).

    nvidia-smi reports the *driver* CUDA version — the ceiling of what the
    GPU driver supports.  PyTorch wheels bundle their own CUDA runtime, so
    you can install a wheel for CUDA 12.4 even if the driver only reports 12.2,
    as long as the bundled runtime is ≤ the driver ceiling.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return None, None

    match = re.search(r"CUDA Version:\s+(\d+)\.(\d+)", result.stdout)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def cuda_version_string() -> Optional[str]:
    """Return CUDA version as 'major.minor' string, or None."""
    major, minor = detect_cuda_version()
    if major is not None:
        return f"{major}.{minor}"
    return None


def recommend_torch_tag(cuda_major: int, cuda_minor: int) -> Optional[str]:
    """Return the best matching PyTorch wheel tag for the given CUDA version.

    Picks the highest CUDA tag whose minimum requirement is met by the driver.
    Returns None if no compatible tag is found.
    """
    for (req_major, req_minor), tag in _CUDA_TO_TORCH_TAG:
        if (cuda_major, cuda_minor) >= (req_major, req_minor):
            return tag
    return None


def torch_install_command(tag: str, package: str = "torch") -> str:
    """Return the pip install command for a given wheel tag."""
    return f"pip install {package} --index-url {TORCH_INDEX_BASE}/{tag}"


def check_torch() -> dict:
    """Return a status dict describing the current torch installation.

    Keys
    ----
    installed : bool
    version : str or None
    cuda_available : bool
    cuda_version : str or None   — version torch was compiled against
    device_count : int
    """
    info: dict = {
        "installed": False,
        "version": None,
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
    }
    try:
        import torch  # noqa: PLC0415

        info["installed"] = True
        info["version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        info["device_count"] = torch.cuda.device_count() if info["cuda_available"] else 0
    except ImportError:
        pass
    return info


def check_mdi() -> dict:
    """Return a status dict for the MDI Python bindings.

    pymdi (PyPI) installs as ``import mdi``.  It does not set ``__version__``
    but does expose MDI_MAJOR_VERSION, MDI_MINOR_VERSION, MDI_PATCH_VERSION
    as module-level integer constants.
    """
    info: dict = {"installed": False, "version": None, "file": None}
    try:
        import mdi  # noqa: PLC0415

        info["installed"] = True

        # Build version string from the integer constants that pymdi exposes.
        # (There is no __version__ attribute on the mdi module.)
        major = getattr(mdi, "MDI_MAJOR_VERSION", None)
        minor = getattr(mdi, "MDI_MINOR_VERSION", None)
        patch = getattr(mdi, "MDI_PATCH_VERSION", None)
        if major is not None:
            info["version"] = f"{major}.{minor}.{patch}"
        else:
            info["version"] = "unknown"

        info["file"] = getattr(mdi, "__file__", "unknown")
    except ImportError:
        pass
    return info


def print_environment_report() -> None:
    """Print a human-readable summary of the runtime environment."""
    print("=" * 60)
    print("lammps-mdi environment report")
    print("=" * 60)

    # Python
    print(f"\nPython:  {sys.version.split()[0]}  ({sys.executable})")

    # CUDA (driver)
    cuda_maj, cuda_min = detect_cuda_version()
    if cuda_maj is not None:
        print(f"GPU:     CUDA driver {cuda_maj}.{cuda_min} detected via nvidia-smi")
        tag = recommend_torch_tag(cuda_maj, cuda_min)
        if tag:
            print(f"         Recommended torch wheel tag: {tag}")
            print(f"         {torch_install_command(tag)}")
    else:
        print("GPU:     nvidia-smi not found — no GPU or driver not in PATH")

    # Torch
    t = check_torch()
    if t["installed"]:
        cuda_str = f"  (CUDA {t['cuda_version']}, {t['device_count']} device(s))" \
                   if t["cuda_available"] else "  (CPU-only build)"
        print(f"\ntorch:   {t['version']}{cuda_str}")
    else:
        print("\ntorch:   NOT INSTALLED")
        print("         Install torch before lammps-mdi — see INSTALL.md")

    # MDI
    m = check_mdi()
    if m["installed"]:
        print(f"\npymdi:   {m['version']}  ({m['file']})")
    else:
        print("\npymdi:   NOT INSTALLED")
        print("         pip install pymdi  OR load the MDI environment module")

    # mace-torch
    try:
        import mace  # noqa: PLC0415

        mace_ver = getattr(mace, "__version__", "installed")
        print(f"\nmace:    {mace_ver}")
    except ImportError:
        print("\nmace:    NOT INSTALLED  (pip install mace-torch)")

    # vesin-torch
    try:
        from vesin.torch import NeighborList  # noqa: PLC0415, F401

        print("vesin:   available (GPU neighbor lists enabled)")
    except ImportError:
        print("vesin:   not installed  (pip install vesin-torch)  — CPU fallback will be used")

    # cuEquivariance
    try:
        import cuequivariance  # noqa: PLC0415

        cueq_ver = getattr(cuequivariance, "__version__", "installed")
        print(f"cueq:    {cueq_ver}")
    except ImportError:
        print("cueq:    not installed  (optional, see INSTALL.md)")

    print()
