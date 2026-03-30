"""
lammps-mdi — MDI engine drivers for LAMMPS.

Provides ML forcefield engines that communicate with LAMMPS via the
MolSSI Driver Interface (MDI), allowing GPU-accelerated models such as
MACE to be used with standard LAMMPS builds (no Kokkos required).

Available engines
-----------------
MACEEngine : MACE-torch model server (mace_mdi.py / mace-mdi command)

Planned
-------
NequIPEngine, SevenNetEngine, ...

Shell scripts
-------------
The package bundles CPU/GPU binding and monitoring scripts for HPC use.
Install them with::

    lammps-mdi install-scripts [--dir ~/SEAMM/bin]
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0.dev0"

__all__ = ["MACEEngine", "__version__"]


def __getattr__(name: str):
    """Lazy-import heavy engine classes so that importing lammps_mdi itself
    does not require torch / mdi / mace to be installed.  They are only
    needed at runtime when an engine is actually used."""
    if name == "MACEEngine":
        from .mace_mdi import MACEEngine  # noqa: PLC0415
        return MACEEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
