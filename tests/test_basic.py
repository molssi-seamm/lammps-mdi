"""
Basic tests for lammps-mdi — these do not require a GPU or a MACE model.
They check that the package is importable and that the CLI tools work.
"""

import subprocess
import sys


def test_import():
    """Package should be importable."""
    import lammps_mdi
    assert hasattr(lammps_mdi, "__version__")
    assert hasattr(lammps_mdi, "MACEEngine")


def test_check_mdi_version_constants():
    """check_mdi() should handle mdi's non-standard version exposure."""
    from lammps_mdi.cuda_utils import check_mdi
    # We can't guarantee mdi is installed in the test environment,
    # but if it is, the version string should be built from integer constants.
    info = check_mdi()
    if info["installed"]:
        assert info["version"] is not None
        # Should be "major.minor.patch" or "unknown", never raise
        parts = info["version"].split(".")
        assert len(parts) == 3 or info["version"] == "unknown"


def test_cuda_utils_importable():
    from lammps_mdi.cuda_utils import (
        detect_cuda_version,
        recommend_torch_tag,
        torch_install_command,
        check_torch,
        check_mdi,
    )
    # These should not raise even if no GPU is present
    maj, min_ = detect_cuda_version()
    assert maj is None or isinstance(maj, int)


def test_recommend_torch_tag():
    from lammps_mdi.cuda_utils import recommend_torch_tag
    assert recommend_torch_tag(12, 8) == "cu128"
    assert recommend_torch_tag(12, 2) == "cu121"   # 12.2 >= 12.1
    assert recommend_torch_tag(12, 1) == "cu121"
    assert recommend_torch_tag(11, 8) == "cu118"
    assert recommend_torch_tag(11, 0) is None       # too old


def test_mace_mdi_parse_args():
    """mace-mdi argparse should handle --help without error."""
    from lammps_mdi.mace_mdi import parse_args
    args = parse_args(["-mdi", "-role ENGINE -name MACE -method MPI"])
    assert args.mdi_args == "-role ENGINE -name MACE -method MPI"
    assert args.device is None
    assert args.dtype is None
    assert args.enable_cueq is False


def test_scripts_bundled():
    """All expected shell scripts should be present as package data."""
    import importlib.resources
    scripts = importlib.resources.files("lammps_mdi") / "scripts"
    for name in ["mdi_bind.sh", "mdi_monitor.sh", "cpu_bind.sh", "gpu_bind.sh"]:
        resource = scripts / name
        assert resource.is_file(), f"Missing bundled script: {name}"


def test_cli_version():
    """lammps-mdi version subcommand should succeed."""
    result = subprocess.run(
        [sys.executable, "-m", "lammps_mdi.cli", "version"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "lammps-mdi" in result.stdout
