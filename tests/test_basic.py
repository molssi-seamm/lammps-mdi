"""
Tests for lammps-mdi.

Split into two groups:

  lightweight — no GPU, no torch, no mdi, no mace required.
                These always run in CI.

  gpu/runtime — require torch, mdi, mace etc.  Skipped automatically
                when those packages are not installed (i.e. in CI).
"""

import subprocess
import sys

import pytest

# ---------------------------------------------------------------------------
# Lightweight tests — always run in CI
# ---------------------------------------------------------------------------


def test_package_importable():
    """lammps_mdi itself should import without torch/mdi/mace present."""
    import lammps_mdi

    assert hasattr(lammps_mdi, "__version__")
    # MACEEngine is lazy — accessing it triggers torch import, so we only
    # confirm it is listed in __all__ rather than actually importing it here.
    assert "MACEEngine" in lammps_mdi.__all__


def test_cuda_utils_importable():
    """cuda_utils has no heavy dependencies and should always import."""
    from lammps_mdi.cuda_utils import (
        detect_cuda_version,
        recommend_torch_tag,
        torch_install_command,
        check_torch,
        check_mdi,
    )

    maj, min_ = detect_cuda_version()
    assert maj is None or isinstance(maj, int)


def test_recommend_torch_tag():
    from lammps_mdi.cuda_utils import recommend_torch_tag

    assert recommend_torch_tag(12, 8) == "cu128"
    assert recommend_torch_tag(12, 6) == "cu126"
    assert recommend_torch_tag(12, 4) == "cu124"
    assert recommend_torch_tag(12, 2) == "cu121"  # 12.2 >= 12.1 minimum
    assert recommend_torch_tag(12, 1) == "cu121"
    assert recommend_torch_tag(11, 8) == "cu118"
    assert recommend_torch_tag(11, 0) is None  # too old


def test_torch_install_command_format():
    from lammps_mdi.cuda_utils import torch_install_command

    cmd = torch_install_command("cu121")
    assert "torch" in cmd
    assert "cu121" in cmd
    assert "download.pytorch.org" in cmd


def test_check_torch_no_gpu():
    """check_torch() must not raise when torch is absent."""
    from lammps_mdi.cuda_utils import check_torch

    info = check_torch()
    assert "installed" in info
    assert "cuda_available" in info


def test_check_mdi_version_constants():
    """check_mdi() must handle mdi's non-standard version exposure gracefully."""
    from lammps_mdi.cuda_utils import check_mdi

    info = check_mdi()
    assert "installed" in info
    if info["installed"]:
        # pymdi exposes MDI_MAJOR/MINOR/PATCH_VERSION, not __version__
        assert info["version"] is not None
        parts = info["version"].split(".")
        assert len(parts) == 3 or info["version"] == "unknown"


def test_scripts_bundled():
    """All expected shell scripts must be present as package data."""
    import importlib.resources

    scripts = importlib.resources.files("lammps_mdi") / "scripts"
    for name in ["mdi_bind.sh", "mdi_monitor.sh", "cpu_bind.sh", "gpu_bind.sh"]:
        resource = scripts / name
        assert resource.is_file(), f"Missing bundled script: {name}"


def test_cli_version():
    """lammps-mdi version subcommand should succeed."""
    result = subprocess.run(
        [sys.executable, "-m", "lammps_mdi.cli", "version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "lammps-mdi" in result.stdout


def test_cli_check_runs():
    """lammps-mdi check should always exit 0 (it reports, does not assert)."""
    result = subprocess.run(
        [sys.executable, "-m", "lammps_mdi.cli", "check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


# ---------------------------------------------------------------------------
# Runtime tests — skipped automatically when heavy deps are absent
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="torch not installed — skipping GPU tests")
mdi_mod = pytest.importorskip("mdi", reason="pymdi not installed — skipping MDI tests")


def test_mace_mdi_parse_args():
    """mace-mdi argparse must work when torch and mdi are available."""
    from lammps_mdi.mace_mdi import parse_args

    args = parse_args(["-mdi", "-role ENGINE -name MACE -method MPI"])
    assert args.mdi_args == "-role ENGINE -name MACE -method MPI"
    assert args.device is None
    assert args.dtype is None
    assert args.enable_cueq is False
    assert args.enable_oeq is False
    assert args.log_level == "INFO"


def test_mace_engine_class_importable():
    """MACEEngine class should be importable when torch/mdi are present."""
    from lammps_mdi.mace_mdi import MACEEngine

    assert callable(MACEEngine)


def test_check_torch_with_torch():
    """check_torch() should detect the installed torch correctly."""
    from lammps_mdi.cuda_utils import check_torch

    info = check_torch()
    assert info["installed"] is True
    assert info["version"] is not None
