"""
lammps-mdi — top-level CLI.

Subcommands
-----------
check           Report environment (CUDA, torch, MDI, mace, etc.)
install-scripts Copy bundled shell scripts to a target directory
install-torch   Print the correct pip install command for torch on this machine
"""

import argparse
import importlib.resources
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Subcommand: check
# ---------------------------------------------------------------------------


def cmd_check(args) -> None:
    from .cuda_utils import print_environment_report

    print_environment_report()


# ---------------------------------------------------------------------------
# Subcommand: install-scripts
# ---------------------------------------------------------------------------

_SCRIPTS = [
    "mdi_bind.sh",
    "mdi_monitor.sh",
    "cpu_bind.sh",
    "gpu_bind.sh",
]


def cmd_install_scripts(args) -> None:
    target = Path(args.dir).expanduser()
    target.mkdir(parents=True, exist_ok=True)

    pkg_scripts = importlib.resources.files("lammps_mdi") / "scripts"

    installed = []
    for script_name in _SCRIPTS:
        src = pkg_scripts / script_name
        dst = target / script_name
        if not importlib.resources.is_resource("lammps_mdi.scripts", script_name):
            # Fall back: try reading via files() API (Python ≥ 3.9)
            pass
        try:
            content = src.read_bytes()
        except Exception as e:
            print(f"Warning: could not read {script_name}: {e}", file=sys.stderr)
            continue

        dst.write_bytes(content)
        dst.chmod(dst.stat().st_mode | 0o111)  # ensure executable
        installed.append(dst)
        print(f"  Installed: {dst}")

    if installed:
        print(f"\n{len(installed)} script(s) installed to {target}")
        print("Make sure that directory is in your PATH, or use full paths in lammps.ini")
    else:
        print("No scripts were installed (check warnings above).", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand: install-torch
# ---------------------------------------------------------------------------


def cmd_install_torch(args) -> None:
    from .cuda_utils import (
        detect_cuda_version,
        recommend_torch_tag,
        torch_install_command,
    )

    major, minor = detect_cuda_version()
    if major is None:
        print("No GPU detected via nvidia-smi.")
        print("For CPU-only torch:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
        return

    print(f"Detected CUDA driver version: {major}.{minor}")
    tag = recommend_torch_tag(major, minor)
    if tag is None:
        print(
            f"CUDA {major}.{minor} is older than the minimum supported by recent PyTorch (11.8).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Recommended PyTorch wheel: {tag}")
    print()
    print("Run:")
    print(f"  {torch_install_command(tag)}")
    print()
    print("Note: PyTorch wheels bundle their own CUDA runtime libraries.")
    print("The wheel version can be slightly newer than the driver CUDA ceiling;")
    print("as long as the driver version >= the wheel's minimum, it will work.")
    print()
    print("After installing torch, install lammps-mdi:")
    print("  pip install lammps-mdi[gpu]          # with vesin-torch (recommended)")
    print("  pip install lammps-mdi[gpu-full]     # also adds cuEquivariance base packages")
    print()
    print("For cuEquivariance ops (choose cu11 or cu12 to match your driver):")
    print("  pip install cuequivariance-ops-torch-cu12")


# ---------------------------------------------------------------------------
# Subcommand: version
# ---------------------------------------------------------------------------


def cmd_version(args) -> None:
    from . import __version__

    print(f"lammps-mdi {__version__}")


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="lammps-mdi",
        description="lammps-mdi — MDI engine drivers for LAMMPS",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # check
    p_check = sub.add_parser("check", help="Report the runtime environment")
    p_check.set_defaults(func=cmd_check)

    # install-scripts
    p_scripts = sub.add_parser(
        "install-scripts",
        help="Install bundled shell scripts (mdi_bind.sh, etc.) to a directory",
    )
    p_scripts.add_argument(
        "--dir",
        "-d",
        default="~/SEAMM/bin",
        metavar="DIR",
        help="Target directory (default: ~/SEAMM/bin)",
    )
    p_scripts.set_defaults(func=cmd_install_scripts)

    # install-torch
    p_torch = sub.add_parser(
        "install-torch",
        help="Print the correct pip install command for torch on this machine",
    )
    p_torch.set_defaults(func=cmd_install_torch)

    # version
    p_ver = sub.add_parser("version", help="Print the lammps-mdi version")
    p_ver.set_defaults(func=cmd_version)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
