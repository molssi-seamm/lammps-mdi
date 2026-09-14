"""ml_install.py — install the PyTorch/MACE stack for a machine.

Backs the ``lammps-mdi install-ml`` command.

Getting this stack right by hand is easy to get wrong in ways that fail late
and confusingly, so the ordering and index choices here are deliberate:

* **The torch wheel depends on the machine's driver, not on preference.**  A
  ``cu130`` wheel on a driver that reports CUDA 12.2 imports fine and then
  reports ``torch.cuda.is_available() == False``, so the first thing that
  actually touches the GPU fails -- for MACE, a ``torch.load`` of a
  GPU-serialised model, nowhere near the real cause.  CUDA *minor* version
  compatibility means any 12.x wheel runs on a 12.1+ driver; a major bump
  does not.

* **Never let PyPI resolve torch.**  ``mace-torch`` depends on torch, so a
  plain ``pip install mace-torch`` pulls whatever the default index calls
  newest -- currently a CUDA 13 build.  Every command here therefore runs
  with the CUDA-specific PyTorch index as ``--index-url`` and PyPI only as
  ``--extra-index-url``, so anything that pulls torch in still gets the build
  this machine can run.

* **Resolve torch and vesin-torch together.**  vesin-torch caps the torch
  version it supports (``torch<2.14`` at the time of writing).  Installed
  afterwards it would drag torch back down -- off PyPI, undoing the point.
  Installed in the same command, pip picks a consistent pair from the CUDA
  index instead, and the cap is honoured without pinning a version here that
  would go stale.

* **``vesin`` and ``vesin-torch`` are separate packages.**  ``vesin.torch``
  lives in the ``vesin`` namespace but is provided by ``vesin-torch``;
  installing only the latter leaves ``import vesin`` failing.
"""

import platform
import shutil
import subprocess
import sys
from typing import NamedTuple, Optional

from .cuda_utils import TORCH_INDEX_BASE, detect_cuda_version, recommend_torch_tag

PYPI_INDEX = "https://pypi.org/simple"


class Target(NamedTuple):
    """What this machine can run."""

    kind: str  # "cuda" | "cpu" | "mac"
    tag: Optional[str]  # torch wheel tag, e.g. "cu126"; None on mac
    index_url: Optional[str]  # torch index; None means plain PyPI
    detail: str  # human-readable reason


class Step(NamedTuple):
    """One pip invocation."""

    description: str
    args: list


def cueq_ops_package(tag: str) -> str:
    """Return the cuEquivariance ops package matching a torch wheel tag.

    The ops kernels are built per CUDA major version, so the choice follows
    the wheel tag rather than the driver.
    """
    return (
        "cuequivariance-ops-torch-cu11"
        if tag.startswith("cu11")
        else ("cuequivariance-ops-torch-cu12")
    )


def resolve_target(tag_override: Optional[str] = None) -> Target:
    """Work out which torch build this machine needs.

    ``tag_override`` forces a wheel tag.  The automatic choice is the newest
    tag the driver supports, but a newer *tag* does not imply a newer *torch*:
    PyTorch stops building the older CUDA variants for its newest releases and
    only starts building the newest variant part-way through a series, so for a
    given Python version one tag can offer a materially newer torch than
    another that is equally runnable.  Override when that matters -- or to
    avoid changing a working installation.
    """
    if tag_override:
        return Target(
            kind="cpu" if tag_override == "cpu" else "cuda",
            tag=tag_override,
            index_url=f"{TORCH_INDEX_BASE}/{tag_override}",
            detail=f"{tag_override} requested explicitly",
        )

    if platform.system() == "Darwin":
        return Target(
            kind="mac",
            tag=None,
            index_url=None,
            detail="macOS — the default PyPI wheel carries MPS support",
        )

    major, minor = detect_cuda_version()
    if major is None:
        return Target(
            kind="cpu",
            tag="cpu",
            index_url=f"{TORCH_INDEX_BASE}/cpu",
            detail="no NVIDIA driver found via nvidia-smi — installing the CPU build",
        )

    tag = recommend_torch_tag(major, minor)
    if tag is None:
        raise RuntimeError(
            f"driver reports CUDA {major}.{minor}, older than the minimum supported "
            "by recent PyTorch (11.8). Update the NVIDIA driver."
        )
    return Target(
        kind="cuda",
        tag=tag,
        index_url=f"{TORCH_INDEX_BASE}/{tag}",
        detail=f"driver reports CUDA {major}.{minor} — using the {tag} wheels",
    )


def build_plan(target: Target, with_cueq: bool = True, with_vesin: bool = True) -> list:
    """Return the ordered pip steps for this target.

    Pure function -- it runs nothing, so the plan can be shown or tested.
    """
    if target.index_url:
        index = ["--index-url", target.index_url, "--extra-index-url", PYPI_INDEX]
    else:
        index = []

    steps = []

    # torch (+ vesin) first, resolved together so vesin-torch's cap on torch is
    # satisfied from the CUDA index rather than from PyPI.
    first = ["torch"]
    if with_vesin:
        first += ["vesin", "vesin-torch"]
    steps.append(Step(f"PyTorch{' and vesin' if with_vesin else ''}", index + first))

    # cuEquivariance only makes sense with real CUDA kernels.
    if with_cueq and target.kind == "cuda":
        steps.append(
            Step(
                "cuEquivariance",
                index + ["cuequivariance", "cuequivariance-torch", cueq_ops_package(target.tag)],
            )
        )

    # MACE and the rest of this package's runtime deps.  Already present for a
    # normal `pip install lammps-mdi`; needed after a --no-deps install, and
    # harmless either way.  Kept behind the same index so a torch touch here
    # cannot substitute a wheel this machine cannot run.
    steps.append(Step("MACE and support packages", index + ["mace-torch", "matscipy", "pint"]))

    return steps


def format_plan(target: Target, steps: list) -> str:
    """Render the plan as the commands that would run."""
    lines = [f"Target: {target.detail}", ""]
    for n, step in enumerate(steps, 1):
        lines.append(f"{n}. {step.description}")
        lines.append("   pip install " + " ".join(step.args))
    return "\n".join(lines)


def verify() -> list:
    """Import-check the stack; return a list of human-readable result lines."""
    results = []

    try:
        import torch  # noqa: PLC0415

        cuda = torch.cuda.is_available()
        line = f"torch {torch.__version__}"
        if torch.version.cuda:
            line += f" (built for CUDA {torch.version.cuda})"
        line += f" — cuda available: {cuda}"
        if cuda:
            line += f", {torch.cuda.device_count()} device(s)"
        results.append(line)
        if torch.version.cuda and not cuda and platform.system() != "Darwin":
            results.append(
                "  WARNING: torch has a CUDA build but cannot use it. The driver is "
                "probably older than the wheel needs — re-run `lammps-mdi check`."
            )
    except ImportError as e:
        results.append(f"torch: NOT importable ({e})")

    for mod, label in (
        ("vesin", "vesin"),
        ("vesin_torch", "vesin-torch"),
        ("cuequivariance", "cuEquivariance"),
        ("cuequivariance_ops_torch", "cuEquivariance ops"),
        ("mace", "mace-torch"),
    ):
        try:
            __import__(mod)
            results.append(f"{label}: OK")
        except ImportError:
            results.append(f"{label}: not installed")

    return results


def install_ml(
    dry_run: bool = False,
    with_cueq: bool = True,
    with_vesin: bool = True,
    tag: Optional[str] = None,
) -> int:
    """Install the ML stack. Returns a process exit code."""
    try:
        target = resolve_target(tag)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    steps = build_plan(target, with_cueq=with_cueq, with_vesin=with_vesin)

    print(format_plan(target, steps))
    print()

    if dry_run:
        print("Dry run — nothing was installed.")
        return 0

    if shutil.which(sys.executable) is None:  # pragma: no cover - defensive
        print(f"Error: cannot find the running interpreter {sys.executable}", file=sys.stderr)
        return 1

    for n, step in enumerate(steps, 1):
        print(f"--- [{n}/{len(steps)}] {step.description} ---", flush=True)
        cmd = [sys.executable, "-m", "pip", "install"] + step.args
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"\nError: step {n} ({step.description}) failed with exit "
                f"{result.returncode}. Nothing after it was attempted.",
                file=sys.stderr,
            )
            return result.returncode

    print("\n--- verifying ---")
    for line in verify():
        print(f"  {line}")
    print("\nDone. `lammps-mdi check` shows the full environment report.")
    return 0
