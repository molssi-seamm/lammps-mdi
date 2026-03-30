# History

All notable changes to **lammps-mdi** are recorded here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-03-30

First public release.

### Added

- `MACEEngine`: MDI engine that runs MACE-torch models, communicating
  with a LAMMPS driver process via MPI-MDI.  Bypasses the ASE Calculator
  interface for lower per-step overhead, building model inputs directly
  from MDI data.
- Automatic selection between **vesin-torch** (GPU-accelerated) and
  **matscipy** (CPU fallback) neighbor list backends.
- Optional **cuEquivariance** (`--enable-cueq`) and
  **openEquivariance** (`--enable-oeq`) acceleration for MACE.
- All heavy runtime dependencies (`torch`, `mdi`, `mpi4py`, `mace`,
  `numpy`, `vesin`, `matscipy`) are imported lazily so the package is
  importable on any machine without a GPU.
- Bundled shell scripts for CPU/GPU resource binding on HPC clusters:
  - `mdi_bind.sh` — binds engine (rank 0) to GPU + NUMA-local CPUs and
    driver (rank 1) to adjacent CPUs; starts nvidia-smi monitor.
    For standalone machines.
  - `mdi_monitor.sh` — lightweight wrapper for SLURM/PBS managed
    environments; scheduler handles binding, script adds GPU monitoring.
  - `gpu_bind.sh` — per-rank GPU binding for native Kokkos LAMMPS.
  - `cpu_bind.sh` — CPU-only binding using L3 cache groups (EPYC 7763).
- `lammps-mdi` CLI with subcommands:
  - `check` — report the runtime environment (CUDA, torch, MDI, mace, …)
  - `install-scripts` — copy bundled shell scripts to a target directory
  - `install-torch` — print the correct `pip install torch` command for
    the detected CUDA driver version
  - `version` — print the installed package version
- `mace-mdi` console script as the entry point for the MACE MDI engine.
- CUDA auto-detection via `nvidia-smi`; maps driver CUDA version to the
  appropriate PyTorch wheel tag (cu118 … cu128).
- BSD-3-Clause license.

---

## [Unreleased]

### Planned
- NequIPEngine: MDI engine for NequIP/Allegro models.
- SevenNetEngine: MDI engine for SevenNet models.
- Configurable CPU topology in binding scripts (currently hard-coded for
  dual-GPU EPYC 7763).

---

[0.1.0]: https://github.com/molssi-seamm/lammps-mdi/releases/tag/v0.1.0
[Unreleased]: https://github.com/molssi-seamm/lammps-mdi/compare/v0.1.0...HEAD
