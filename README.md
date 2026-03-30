# lammps-mdi

MDI engine drivers for LAMMPS — run ML forcefields (MACE, and future models)
on GPU via the [MolSSI Driver Interface](https://molssi-mdi.github.io/MDI_Library/),
communicating with a standard LAMMPS binary (no Kokkos compilation needed).

Designed for use with [SEAMM](https://github.com/molssi-seamm), but works
with any LAMMPS workflow that supports MDI.

## How it works

LAMMPS acts as an MDI **driver**: it handles atom positions, neighbor lists
(at the coarse level), and time integration.  The lammps-mdi engine process
acts as an MDI **engine**: it receives coordinates from LAMMPS each step,
evaluates the ML model on GPU, and returns energies, forces, and (if periodic)
the stress tensor.

```
mpirun -np 1  mace-mdi  -mdi "..."   ← GPU process: MACE on A100
         : -np 1  lmp  -mdi "..." -in input.dat  ← CPU process: time integration
```

The two processes communicate over MPI via the MDI protocol.

## Supported engines

| Engine     | Status | Notes |
|------------|--------|-------|
| MACE       | ✅     | MACE-torch ≥ 0.3, vesin-torch neighbor lists, cuEquivariance optional |
| NequIP     | Planned | |
| SevenNet   | Planned | |

## Installation

See [INSTALL.md](INSTALL.md) for full HPC instructions.  Short version:

```bash
# 1. Load your LAMMPS module (provides Python, numpy, mpi4py, MDI)
module load LAMMPS/...

# 2. Create a venv that inherits the module stack
python -m venv --system-site-packages ~/venvs/lammps-mdi
source ~/venvs/lammps-mdi/bin/activate

# 3. Install PyTorch with the right CUDA wheel (check your CUDA version first)
lammps-mdi install-torch    # prints the correct command
# e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 4. Install lammps-mdi
pip install lammps-mdi[gpu]

# 5. Install bundled shell scripts
lammps-mdi install-scripts

# 6. Verify
lammps-mdi check
```

## Usage

### As a console script (recommended)

```bash
SEAMM_FF=/path/to/model.model \
mpirun --mca mpi_yield_when_idle 1 \
    -np 1 mdi_bind.sh mace-mdi -mdi "-role ENGINE -name MACE -method MPI" \
    : -np 1 mdi_bind.sh lmp -mdi "-role DRIVER -name LAMMPS -method MPI" -in input.dat
```

The `mace-mdi` command accepts several options:

```
mace-mdi --help

  -mdi MDI_STRING      MDI initialization string [required]
  --model PATH         Path to MACE model (overrides SEAMM_FF)
  --device DEVICE      PyTorch device (default: cuda:0)
  --dtype {float32,float64}
  --enable-cueq        Enable cuEquivariance acceleration
  --enable-oeq         Enable openEquivariance acceleration
  --log-level LEVEL    DEBUG / INFO / WARNING / ERROR
```

### From lammps.ini (SEAMM)

```ini
[local]
installation = conda   # or modules, or local

gpu-code = mpirun --mca mpi_yield_when_idle 1 \
    -np 1 ~/SEAMM/bin/mdi_bind.sh \
    mace-mdi -mdi "-role ENGINE -name MACE -method MPI" \
    : -np 1 ~/SEAMM/bin/mdi_bind.sh \
    lmp -mdi "-role DRIVER -name LAMMPS -method MPI"
```

### As a Python library

```python
from lammps_mdi import MACEEngine

engine = MACEEngine(
    model_path="/path/to/model.model",
    device="cuda:0",
    default_dtype="float32",
    enable_cueq=True,
)
engine.run("-role ENGINE -name MACE -method MPI")
```

## Shell scripts

The package bundles four helper scripts, installed via `lammps-mdi install-scripts`:

| Script | Purpose |
|--------|---------|
| `mdi_bind.sh` | Binds engine (rank 0) to GPU + NUMA-local CPUs, driver (rank 1) to adjacent CPUs; starts nvidia-smi monitor. For standalone machines. |
| `mdi_monitor.sh` | Lightweight wrapper for SLURM/PBS: only starts GPU monitoring. Scheduler handles binding. |
| `gpu_bind.sh` | Per-rank GPU binding for native Kokkos LAMMPS (approach A). |
| `cpu_bind.sh` | CPU-only binding using L3 cache groups (EPYC 7763). |

The CPU/GPU mappings in `mdi_bind.sh`, `gpu_bind.sh`, and `cpu_bind.sh` are
currently hard-coded for a dual-GPU EPYC 7763 system.  They will be made
configurable in a future release.

## Requirements

| Package | Source | Notes |
|---------|--------|-------|
| Python ≥ 3.10 | HPC module | |
| numpy | HPC module | Do **not** reinstall |
| mpi4py | HPC module | |
| pymdi ≥ 1.4 | pip | PyPI package for `import mdi` |
| torch (CUDA) | pip (special index) | Install before lammps-mdi |
| mace-torch ≥ 0.3 | pip | |
| matscipy ≥ 0.8 | pip | CPU fallback neighbor list |
| pint ≥ 0.20 | pip | Unit conversion |
| vesin-torch ≥ 0.3 | pip (optional) | GPU neighbor lists, strongly recommended |
| cuequivariance* | pip (optional) | NVIDIA cuEquivariance acceleration |

## Contributing

Issues and pull requests are welcome at
https://github.com/molssi-seamm/lammps-mdi.

## License

MIT — see [LICENSE](LICENSE).
