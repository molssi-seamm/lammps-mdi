# History

All notable changes to **lammps-mdi** are recorded here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.12] — 2026-09-15

### Fixed
- `mdi_bind.sh` no longer puts two concurrent jobs on the same cores. It picks
  the cores for the engine and the driver from a map of the machine's GPUs, so
  that each runs near its card, but looked that map up by the GPU's index
  *within the job's own allocation* — which is 0 for every job given a single
  GPU, whichever card it holds. Two jobs running at once therefore both took
  GPU 0's cores and shared them, at about half a CPU each rather than one
  each, even though the scheduler had correctly given them different cards. It
  now looks up by the physical index, which is what the map is written in
  terms of. A job with no scheduler allocation is unchanged.

---

## [0.1.11] — 2026-09-14

### Fixed
- A failure while building the engine now aborts the MPI job instead of
  hanging. Loading the model is where a run is most likely to die -- a model
  pickled against a module the environment does not have, a file that will not
  deserialise, a GPU with too little memory -- and none of it was covered by
  the guard inside `run()`, because MDI has not been initialised that early.
  LAMMPS was left waiting for an engine that would never answer, so the job
  burned its whole wall-clock allocation before anyone saw the traceback. Seen
  for real as `ModuleNotFoundError: No module named 'xnns'` from a model with
  an unsatisfied dependency: the engine died, and the job sat allocated until
  it was cancelled by hand. It now exits in seconds, naming the model path.

---

## [0.1.10] — 2026-09-14

### Added
- `lammps-mdi install-ml` installs the ML runtime in one step: PyTorch built
  for this machine's NVIDIA driver, then vesin, cuEquivariance and MACE.
  Getting this right by hand is easy to get wrong in ways that fail late — a
  `cu130` wheel on a CUDA 12.2 driver imports fine and only fails when
  something first touches the GPU, and `pip install mace-torch` pulls torch
  from PyPI, which is exactly how the wrong build arrives. Every step runs
  with the CUDA-specific PyTorch index, and torch and vesin-torch are resolved
  together so vesin's cap on torch is met from that index rather than by
  silently substituting a PyPI build. `--dry-run` shows the plan, `--tag`
  forces a wheel tag, and `--no-cueq` / `--no-vesin` skip the optional pieces.

### Changed
- `make install` installs the package alone, as CI already did, instead of
  pulling the `gpu` extra. It was dragging torch, vesin-torch and MACE into
  whichever environment happened to be active — including, via `make update`,
  a SEAMM development environment as a side effect of making a release. The
  runtime stack belongs in the LAMMPS environment; `install-ml` puts it there.

---

## [0.1.9] — 2026-09-14

### Added
- `--profile-steps N` on `mace-mdi`: profiles the first N forward passes with
  `torch.profiler` and writes a Chrome trace (view at `chrome://tracing` or
  <https://ui.perfetto.dev>).
- The periodic timing line now reports the largest pairs-per-atom seen against
  the `VESIN_CUDA_MAX_PAIRS_PER_POINT` limit, so an undersized
  `--max-pairs-per-point` is visible while the run is going rather than only
  when it fails.
- `lammps-mdi install-mdi` now works on macOS: it builds and inspects
  `libmdi.dylib` with `otool`, instead of assuming `libmdi.so` and `ldd`.

### Fixed
- A MACE model saved on CUDA now loads on a machine with no usable GPU. Such
  models carry e3nn JIT submodules whose `__setstate__` calls `torch.jit.load`
  without a `map_location`, baking CUDA device references into the bytecode,
  which the outer `torch.load(map_location="cpu")` never reached; loading died
  with "Could not run 'aten::empty_strided' with CUDA backend".
- `mdi_bind.sh` no longer reports failure after a successful run. Its engine
  branch kills the `nvidia-smi` monitor it started, and the status of those
  deliberate `kill`/`wait` calls became the script's own, so rank 0 exited 1
  even when the engine finished cleanly and `mpirun` reported the whole run as
  failed. A genuine failure in either branch still propagates.
- An unhandled MDI command now aborts the MPI job instead of being ignored,
  which left LAMMPS waiting for a reply that never came until its wall-clock
  limit expired.
- The engine runs on CPU and Apple MPS, not just CUDA: synchronisation and
  cache teardown are dispatched per device, and vesin's GPU neighbour lists
  are used only on CUDA, falling back to matscipy elsewhere.
- Model timing is measured after synchronising the device. GPU work is
  asynchronous, so the previous figure recorded when the kernel was *launched*
  rather than when it finished; the time spent synchronising and extracting
  results is now reported separately.

### Note
Releases 0.1.1 through 0.1.8 are not recorded in this file.

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

[0.1.12]: https://github.com/molssi-seamm/lammps-mdi/releases/tag/0.1.12
[0.1.11]: https://github.com/molssi-seamm/lammps-mdi/releases/tag/0.1.11
[0.1.10]: https://github.com/molssi-seamm/lammps-mdi/releases/tag/0.1.10
[0.1.9]: https://github.com/molssi-seamm/lammps-mdi/releases/tag/0.1.9
[0.1.0]: https://github.com/molssi-seamm/lammps-mdi/releases/tag/v0.1.0
[Unreleased]: https://github.com/molssi-seamm/lammps-mdi/compare/0.1.12...HEAD
