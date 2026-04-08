# Installing lammps-mdi with Conda

This is currently the recommended installation path. It uses a conda
environment to provide LAMMPS, OpenMPI, pymdi, Python, and numpy, with
pip providing the ML stack (torch, mace, vesin, cuequivariance).

---

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- An NVIDIA GPU with CUDA driver ≥ 12.1
- A local build of LAMMPS with MDI and OpenMPI support (from the
  molssi-seamm conda channel or conda-forge)

Check your CUDA driver version:
```bash
nvidia-smi | grep "CUDA Version"
```

---

## Step 1 — Create the conda environment with LAMMPS

```bash
# Let conda choose the MPI variant (usually OpenMPI by default):
conda create -n seamm-lammps -c conda-forge lammps python=3.12

# Or explicitly request the OpenMPI variant (recommended for reproducibility):
conda create -n seamm-lammps -c conda-forge "lammps=*=*openmpi*" python=3.12
```

`lammps` will automatically pull in `pymdi`, `mpi4py`, and OpenMPI as
dependencies — no need to list them separately. Specifying `openmpi` as
a separate package would not constrain the LAMMPS build variant and could
result in a mismatched environment.

> **Note on pymdi SafetyError:** You may see a message like:
> ```
> SafetyError: The package for pymdi located at ...
> appears to be corrupted. The path 'lib/libmdi.so.1' has an incorrect size.
> ```
> This is a corrupted conda package cache. Fix it by clearing the cache
> and retrying:
> ```bash
> conda clean --packages --yes
> conda install -n seamm-lammps pymdi
> ```

> **Note on UCX CUDA support:** You will see a message suggesting
> `conda install cuda-cudart cuda-version=12`. This is **not needed**
> for the MDI workflow — UCX CUDA-Direct is only relevant for
> multi-node GPU communication, which is not used here.

> **Note on OpenMPI CUDA awareness:** You may see a message about setting
> `OMPI_MCA_opal_cuda_support=true`. This is also **not needed** —
> MDI passes data between CPU buffers only; GPU operations are internal
> to the Python engine process.

---

## Step 2 — Activate the environment

```bash
conda activate seamm-lammps
```

---

## Step 3 — Install lammps-mdi with GPU support

```bash
pip install lammps-mdi[gpu]
```

This installs torch 2.10.0+cu128, mace-torch, matscipy, vesin-torch,
pint, and lammps-mdi itself. It will also install the base cuequivariance
packages if you use `lammps-mdi[gpu-full]` instead.

> **Note on numpy:** pip may downgrade numpy from conda-forge's 2.x to
> 1.x due to the `numpy<2` constraint in lammps-mdi. This is intentional
> for EasyBuild environments where LAMMPS was compiled against numpy 1.x.
> In a pure conda environment where all packages are compiled against
> numpy 2.x, you can restore it afterwards:
> ```bash
> conda install numpy  # restores conda-forge's numpy version
> ```

---

## Step 4 — Fix the MDI library

This step is required regardless of how pymdi was installed. The conda-forge
LAMMPS package ships `lib/libmdi.so` compiled **without MPI support** (for
use with TCP-only MDI). For MPI mode, it must be replaced with an
MPI-enabled build. The `lammps-mdi install-mdi` command handles this:

```bash
lammps-mdi install-mdi
```

This command:
1. Clones `https://github.com/MolSSI-MDI/MDI_Library`
2. Builds it with `--no-build-isolation` so it uses the active
   environment's OpenMPI (not an isolated environment that might find
   system MPICH)
3. Copies `libmdi.so`, `libmdi.so.1`, and `mdi_name` into
   `site-packages/mdi/` for the Python engine
4. Copies `libmdi.so` and `libmdi.so.1` into `$CONDA_PREFIX/lib/`
   to replace the MPI-less LAMMPS stub

> **Background:** The PyPI `pymdi` wheel bundles `libmdi.so` compiled
> against system MPICH at `/usr/local/lib/libmpi.so.12`. Using this with
> an OpenMPI environment causes "Invalid communicator" MPI errors.
> The conda-forge LAMMPS package ships `lib/libmdi.so` without MPI,
> causing "Failed to initialize MPI" errors. Building from source with
> the active OpenMPI fixes both problems.
>
> The MDI developer is working on MPI-variant conda-forge packages
> (analogous to the LAMMPS `mpi_openmpi` / `mpi_mpich` build strings)
> which will eventually make this step unnecessary.

---

## Step 5 — Install cuEquivariance ops kernel (optional)

The base cuequivariance packages are installed by `lammps-mdi[gpu-full]`.
The compiled ops kernel (which provides additional GPU acceleration for
MACE) requires a separate install from NVIDIA's package index:

```bash
pip install --extra-index-url https://pypi.nvidia.com/ \
    cuequivariance-ops-torch-cu12
```

> **Note:** This downloads from `https://pypi.nvidia.com/`. On systems
> with SSL inspection you may need to add `--trusted-host pypi.nvidia.com`.
>
> The ops kernel requires `nvidia-cublas-cu12 >= 12.5`, which is bundled
> with the cu128 torch wheel. If you see a version conflict, check that
> torch was installed from the cu128 wheel.

---

## Step 6 — Install shell scripts

The package bundles CPU/GPU binding scripts for standalone machines and
HPC clusters:

```bash
lammps-mdi install-scripts           # installs to ~/SEAMM/bin
lammps-mdi install-scripts --dir /path/to/bin  # custom location
```

Installed scripts:

| Script | Purpose |
|--------|---------|
| `mdi_bind.sh` | Binds engine (rank 0) to GPU + NUMA-local CPUs, driver (rank 1) to adjacent CPUs; starts nvidia-smi monitor. For standalone machines. |
| `mdi_monitor.sh` | Lightweight wrapper for SLURM/PBS; scheduler handles binding. |
| `gpu_bind.sh` | Per-rank GPU binding for Kokkos LAMMPS. |
| `cpu_bind.sh` | CPU-only binding (EPYC 7763 topology). |

> **Note:** `mdi_bind.sh`, `gpu_bind.sh`, and `cpu_bind.sh` contain
> CPU/GPU topology hard-coded for a dual-GPU EPYC 7763 system. Edit
> them for your hardware before use.

---

## Step 7 — Verify the installation

```bash
lammps-mdi check
```

Expected output:
```
============================================================
lammps-mdi environment report
============================================================

Python:  3.12.13  (/home/.../envs/seamm-lammps/bin/python3.12)
GPU:     CUDA driver 12.2 detected via nvidia-smi
         Recommended torch wheel tag: cu128
         pip install torch --index-url .../cu128

torch:   2.10.0+cu128  (CUDA 12.8, 2 device(s))

pymdi:   1.4.37  (.../site-packages/mdi/__init__.py)  OK

mace:    0.3.15
vesin:   available (GPU neighbor lists enabled)
cueq:    0.9.1
```

---

## Step 8 — Test MDI connectivity

Before running LAMMPS, verify that MDI MPI communication works:

```bash
# Create test scripts
cat > test_mdi_engine.py << 'EOF'
from mpi4py import MPI
import mdi, sys
mdi.MDI_Init("-role ENGINE -name TEST -method MPI", MPI.COMM_WORLD)
comm = mdi.MDI_Accept_Communicator()
cmd = mdi.MDI_Recv_Command(comm)
print(f"Engine: received command '{cmd}'", flush=True)
sys.exit(0)
EOF

cat > test_mdi_driver.py << 'EOF'
from mpi4py import MPI
import mdi, sys
mdi.MDI_Init("-role DRIVER -name DRIVER -method MPI", MPI.COMM_WORLD)
comm = mdi.MDI_Accept_Communicator()
mdi.MDI_Send_Command("EXIT", comm)
print("Driver: sent EXIT, done.", flush=True)
sys.exit(0)
EOF

# Run the test
mpirun --mca mpi_yield_when_idle 1 \
    -np 1 python test_mdi_engine.py \
    : -np 1 python test_mdi_driver.py
```

Expected output:
```
Driver: sent EXIT, done.
Engine: received command 'EXIT'
```

If this fails, the most likely cause is that `lammps-mdi install-mdi`
linked against the wrong MPI. Check:
```bash
ldd $(python -c "import mdi, os; print(os.path.dirname(mdi.__file__))")/libmdi.so \
    | grep libmpi
# Must show the conda-forge libmpi.so, not /usr/local/lib/libmpi.so.12
```

---

## Step 9 — Run a LAMMPS calculation

Set the path to your MACE model and run:

```bash
export SEAMM_FF=/path/to/your/model.mace.pt

mpirun --mca mpi_yield_when_idle 1 \
    -np 1 ~/SEAMM/bin/mdi_bind.sh \
    mace-mdi -mdi "-role ENGINE -name MACE -method MPI" \
    : -np 1 ~/SEAMM/bin/mdi_bind.sh \
    lmp -mdi "-role DRIVER -name LAMMPS -method MPI" -in input.dat
```

The `mace-mdi` console script is installed into the conda environment's
`bin/` and accepts these options:

```
mace-mdi --help

  -mdi MDI_STRING          MDI initialization string [required]
  --model PATH             Path to MACE model (overrides SEAMM_FF)
  --device DEVICE          PyTorch device (default: cuda:0)
  --dtype {float32,float64}
  --enable-cueq            Enable cuEquivariance acceleration
  --enable-oeq             Enable openEquivariance acceleration
  --log-level LEVEL        DEBUG / INFO / WARNING / ERROR
```

---

## Adding to job scripts (SLURM/PBS)

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2

conda activate seamm-lammps
export SEAMM_FF=/path/to/model.mace.pt

mpirun --mca mpi_yield_when_idle 1 \
    -np 1 ~/SEAMM/bin/mdi_monitor.sh \
    mace-mdi -mdi "-role ENGINE -name MACE -method MPI" \
    : -np 1 ~/SEAMM/bin/mdi_monitor.sh \
    lmp -mdi "-role DRIVER -name LAMMPS -method MPI" -in input.dat
```

Use `mdi_monitor.sh` rather than `mdi_bind.sh` on HPC clusters — the
scheduler handles CPU/GPU binding; `mdi_monitor.sh` only adds GPU
monitoring without interfering with the scheduler's assignments.

---

## Troubleshooting

**"Invalid communicator" MPI error**
The MDI library linked against the wrong MPI. Run:
```bash
ldd $(python -c "import mdi, os; print(os.path.dirname(mdi.__file__))")/libmdi.so | grep libmpi
```
If it shows `/usr/local/lib/libmpi.so.12` (system MPICH) instead of the
conda-forge library, re-run `lammps-mdi install-mdi`.

**"Failed to initialize MPI" from LAMMPS**
The `lib/libmdi.so` in the conda environment has no MPI support.
Re-run `lammps-mdi install-mdi` — it replaces this stub.

**pymdi SafetyError during conda install**
Corrupted conda package cache. Run:
```bash
conda clean --packages --yes
conda install pymdi
```

**numpy version conflict**
pip may downgrade numpy from 2.x to 1.x. In a conda environment where
all packages target numpy 2.x, restore it with:
```bash
conda install numpy
```

**cuequivariance-ops SSL error**
Your institution's SSL inspection proxy is blocking `pypi.nvidia.com`. Add:
```bash
--trusted-host pypi.nvidia.com
```
to the pip install command.
