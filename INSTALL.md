# Installing lammps-mdi on HPC Systems

lammps-mdi uses the Python environment that ships with your LAMMPS build.
On HPC clusters managed with EasyBuild or Spack, that Python is provided by
a module, and its site-packages must not be modified.  The correct approach
is a virtual environment that **inherits** the module stack.

---

## Quick summary

```
1. Load your LAMMPS module (sets Python, numpy, mpi4py, MDI, etc.)
2. Create a venv with --system-site-packages
3. Activate the venv
4. pip install torch with the right CUDA wheel
5. pip install lammps-mdi[gpu]
6. lammps-mdi install-scripts
```

---

## Step 1 — Load the LAMMPS module

```bash
module load LAMMPS/22Jul2025-foss-2024a-kokkos   # adjust to your site
```

This loads Python, numpy, mpi4py, MDI, and everything else the LAMMPS
build depends on.  All of these will be visible inside the venv we create
in the next step.

Verify MDI is accessible:

```bash
python -c "import mdi; print(mdi.__version__, mdi.__file__)"
```

If this fails, either the MDI module was not loaded or `pymdi` is not
installed in the system Python.  If pymdi is not present, install it after
activating the venv (step 3):

```bash
pip install pymdi
```

---

## Step 2 — Create a virtual environment

```bash
python -m venv --system-site-packages ~/venvs/lammps-mdi
```

The `--system-site-packages` flag makes the venv see all packages provided
by the loaded modules (numpy, mpi4py, MDI, lammps, …) **without** copying
or reinstalling them.

---

## Step 3 — Activate the venv

```bash
source ~/venvs/lammps-mdi/bin/activate
```

Add this to your job script or `~/.bashrc` (after the `module load` line).

---

## Step 4 — Install PyTorch with the correct CUDA wheel

PyTorch must be installed **before** lammps-mdi, because mace-torch depends
on it and pip must find it already present to avoid pulling in a CPU-only
or wrong-CUDA build.

Use `lammps-mdi` to find the right command for your GPU:

```bash
# If lammps-mdi is not installed yet, use the helper directly:
python -c "
from lammps_mdi.cuda_utils import detect_cuda_version, recommend_torch_tag, torch_install_command
maj, min_ = detect_cuda_version()
tag = recommend_torch_tag(maj, min_)
print(torch_install_command(tag))
"
```

Or simply run `nvidia-smi` and look for `CUDA Version: X.Y`, then pick the
closest matching tag from the table below:

| Driver CUDA ceiling | Use this wheel tag |
|--------------------|--------------------|
| ≥ 12.8             | `cu128`            |
| ≥ 12.6             | `cu126`            |
| ≥ 12.4             | `cu124`            |
| ≥ 12.1             | `cu121`            |
| ≥ 11.8             | `cu118`            |

Example for CUDA 12.2 (driver ceiling) — use cu121:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> **Note on bundled CUDA**: modern PyTorch wheels bundle their own CUDA
> runtime libraries (`nvidia-cuda-runtime-cu12`, etc.).  This means the
> wheel's CUDA version can be slightly *newer* than the driver's reported
> CUDA ceiling, as long as the driver version itself is compatible.
> For CUDA 12.2 drivers the cu121 wheel works correctly.

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Step 5 — Install lammps-mdi

```bash
# Recommended: with GPU-accelerated neighbor lists
pip install lammps-mdi[gpu]

# Or with cuEquivariance acceleration (base packages only):
pip install lammps-mdi[gpu-full]

# Then add the ops kernel matching your CUDA:
pip install cuequivariance-ops-torch-cu12    # for CUDA 12.x
# pip install cuequivariance-ops-torch-cu11  # for CUDA 11.x
```

Run the environment check:

```bash
lammps-mdi check
```

Expected output (abbreviated):

```
============================================================
lammps-mdi environment report
============================================================

Python:  3.12.3  (/home/user/venvs/lammps-mdi/bin/python)
GPU:     CUDA driver 12.2 detected via nvidia-smi
         Recommended torch wheel tag: cu121
         pip install torch --index-url .../cu121

torch:   2.x.x  (CUDA 12.1, 2 device(s))

pymdi:   1.4.x  (/path/to/mdi/__init__.py)

mace:    0.3.x
vesin:   available (GPU neighbor lists enabled)
cueq:    0.9.x
```

---

## Step 6 — Install shell scripts

The package bundles `mdi_bind.sh`, `mdi_monitor.sh`, `cpu_bind.sh`, and
`gpu_bind.sh`.  Install them to `~/SEAMM/bin` (or any directory in PATH):

```bash
lammps-mdi install-scripts                 # installs to ~/SEAMM/bin
lammps-mdi install-scripts --dir /opt/bin  # custom directory
```

---

## Step 7 — Configure lammps.ini

Set the `gpu-code` key in `lammps.ini` to use the installed scripts.
For a standalone machine with one GPU:

```ini
gpu-code = mpirun --mca mpi_yield_when_idle 1 \
    -np 1 ~/SEAMM/bin/mdi_bind.sh \
    mace-mdi -mdi "-role ENGINE -name MACE -method MPI" \
    : -np 1 ~/SEAMM/bin/mdi_bind.sh \
    lmp -mdi "-role DRIVER -name LAMMPS -method MPI"
```

For HPC with SLURM (scheduler handles binding):

```ini
gpu-code = mpirun \
    -np 1 ~/SEAMM/bin/mdi_monitor.sh \
    mace-mdi -mdi "-role ENGINE -name MACE -method MPI" \
    : -np 1 ~/SEAMM/bin/mdi_monitor.sh \
    lmp -mdi "-role DRIVER -name LAMMPS -method MPI"
```

Note that `mace-mdi` is now a console script installed into the venv's
`bin/`, so it is available directly by name once the venv is activated.

---

## Adding the venv activation to job scripts

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2

module load LAMMPS/22Jul2025-foss-2024a-kokkos
source ~/venvs/lammps-mdi/bin/activate

# SEAMM or direct mpirun command here
```

---

## Troubleshooting

**`ImportError: No module named 'mdi'`**
- The MDI module is not loaded, or `pymdi` is not installed.
  Check `module list` and try `pip install pymdi`.

**`torch.cuda.is_available()` returns False**
- The torch wheel was installed without GPU support.
  Reinstall with the correct `--index-url`.

**`cuequivariance` not found at runtime**
- Install the base packages: `pip install cuequivariance cuequivariance-torch`
- Then the ops kernel: `pip install cuequivariance-ops-torch-cu12`

**numpy version conflict**
- Never install numpy inside the venv with `--system-site-packages`.
  The system numpy (from the module) takes precedence; a second install
  in the venv can shadow it with a different version.
  If pip warns about a numpy conflict from `mace-torch`, this is usually
  harmless — mace-torch will use the system numpy.
