"""
install_mdi.py — build and install pymdi from source against the active MPI.

The PyPI pymdi wheel bundles libmdi.so compiled against system MPICH.
This causes "Invalid communicator" errors when the environment uses OpenMPI.
conda-forge LAMMPS ships lib/libmdi.so compiled without MPI support at all,
causing "Failed to initialize MPI" errors.

This module builds MDI_Library from source using the environment's MPI,
then installs libmdi.so everywhere it needs to be.

Called via:  lammps-mdi install-mdi
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MDI_REPO = "https://github.com/MolSSI-MDI/MDI_Library"


def _run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> int:
    """Run a command, streaming output to stdout."""
    print(f"  + {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if check and result.returncode != 0:
        print(f"Error: command failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    return result.returncode


def find_site_packages() -> Path:
    """Return the first writable site-packages directory."""
    import site

    for path in site.getsitepackages():
        p = Path(path)
        if p.exists() and os.access(p, os.W_OK):
            return p
    print("Error: no writable site-packages directory found.", file=sys.stderr)
    sys.exit(1)


def find_env_lib() -> Path | None:
    """Return $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib if it exists."""
    for var in ("CONDA_PREFIX", "VIRTUAL_ENV"):
        prefix = os.environ.get(var)
        if prefix:
            lib = Path(prefix) / "lib"
            if lib.exists():
                return lib
    return None


def check_mpi() -> None:
    """Verify that mpi4py can import and show which MPI it uses."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "from mpi4py import MPI; print(MPI.Get_library_version())"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            version_line = result.stdout.strip().split("\n")[0]
            print(f"  MPI: {version_line}")
        else:
            print("  Warning: mpi4py not available — MDI MPI mode will not work.")
    except Exception:
        print("  Warning: could not check mpi4py.")


def install_mdi(clone_dir: Path | None = None, keep_build: bool = False) -> None:
    """Build MDI_Library from source and install libmdi into the active environment.

    Parameters
    ----------
    clone_dir : Path or None
        Where to clone MDI_Library.  Defaults to a temporary directory.
    keep_build : bool
        If True, don't delete the clone directory after installation.
    """
    print("\n=== lammps-mdi: building MDI_Library from source ===\n")
    check_mpi()

    site_packages = find_site_packages()
    mdi_pkg_dir = site_packages / "mdi"
    env_lib = find_env_lib()

    print(f"\n  site-packages : {site_packages}")
    print(f"  mdi package   : {mdi_pkg_dir}")
    print(f"  env lib       : {env_lib or '(not found — skipping)'}")

    # ---- Clone ----
    use_tmp = clone_dir is None
    if use_tmp:
        tmpdir = tempfile.mkdtemp(prefix="mdi_build_")
        clone_dir = Path(tmpdir) / "MDI_Library"
    else:
        clone_dir = Path(clone_dir)

    try:
        if clone_dir.exists():
            print(f"\nUsing existing clone at {clone_dir}")
        else:
            print(f"\nCloning MDI_Library into {clone_dir} ...")
            _run(["git", "clone", MDI_REPO, str(clone_dir)])

        # ---- Build ----
        print("\nBuilding (pip install --no-build-isolation) ...")
        print("NOTE: --no-build-isolation ensures the build uses the active MPI,")
        print("      not an isolated environment that might find system MPICH.\n")
        _run(
            [sys.executable, "-m", "pip", "install", ".", "--no-build-isolation"],
            cwd=clone_dir,
        )

        # ---- Find build artefacts ----
        # pip builds into build/lib.*/ — find the libmdi files there
        build_lib = None
        for candidate in sorted(clone_dir.glob("build/lib.*")):
            if (candidate / "libmdi.so").exists():
                build_lib = candidate
                break

        if build_lib is None:
            print(
                "\nError: could not find libmdi.so in build/ directory.\n"
                "The build may have failed silently. Check the output above.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"\nBuild artefacts found at: {build_lib}")

        # ---- Install into site-packages/mdi/ ----
        print(f"\nInstalling into {mdi_pkg_dir} ...")
        mdi_pkg_dir.mkdir(exist_ok=True)
        for fname in ["libmdi.so", "libmdi.so.1", "mdi_name"]:
            src = build_lib / fname
            if src.exists():
                dst = mdi_pkg_dir / fname
                shutil.copy2(src, dst)
                print(f"  Copied {fname}")
            else:
                print(f"  Warning: {fname} not found in build — skipping")

        # ---- Install into env lib/ (replaces MPI-less LAMMPS stub) ----
        if env_lib is not None:
            print(f"\nInstalling into {env_lib} (replaces MPI-less LAMMPS stub) ...")
            for fname in ["libmdi.so", "libmdi.so.1"]:
                src = build_lib / fname
                if src.exists():
                    dst = env_lib / fname
                    shutil.copy2(src, dst)
                    print(f"  Copied {fname}")

        # ---- Verify ----
        print("\nVerifying installation ...")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """import mdi; print(
                    f"  pymdi {mdi.MDI_MAJOR_VERSION}."
                    f"{mdi.MDI_MINOR_VERSION}.{mdi.MDI_PATCH_VERSION} imported OK"
                )""",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print("  Error: import mdi failed after installation!", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            sys.exit(1)

        # Check libmdi.so links against the right MPI
        ldd_result = subprocess.run(
            ["ldd", str(mdi_pkg_dir / "libmdi.so")],
            capture_output=True,
            text=True,
        )
        mpi_lines = [tmp for tmp in ldd_result.stdout.splitlines() if "libmpi" in tmp]
        if mpi_lines:
            print(f"  libmdi.so MPI: {mpi_lines[0].strip()}")
            if "/usr/local/lib/libmpi" in mpi_lines[0]:
                print(
                    "\n  WARNING: libmdi.so still links against system MPICH!",
                    file=sys.stderr,
                )
                print(
                    "  This means the build used MPICH instead of the environment MPI.",
                    file=sys.stderr,
                )
                print(
                    "  Make sure mpi4py is installed and the MPI module is loaded.",
                    file=sys.stderr,
                )
        else:
            print("  Warning: libmdi.so does not appear to link against any MPI library.")

        print("\n=== MDI_Library installation complete ===\n")

    finally:
        if use_tmp and not keep_build:
            shutil.rmtree(tmpdir, ignore_errors=True)
