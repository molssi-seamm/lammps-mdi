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

| Driver CUDA ceiling | Use this wheel tag | Notes |
|--------------------|-------------------|-------|
| ≥ 12.1             | `cu128`           | Recommended for all CUDA 12.x systems |
| ≥ 11.8             | `cu118`           | CUDA 11.x systems only |

Modern PyTorch wheels **bundle all their own CUDA runtime libraries**
(`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, etc.) and do not use the
system CUDA installation at all.  A cu128 wheel works correctly on a
system whose driver reports CUDA 12.2, because the driver ABI is stable
across minor CUDA versions.

This matters for compatibility with `cuequivariance-ops-torch-cu12`, which
requires `nvidia-cublas-cu12 >= 12.5`.  The cu128 wheel bundles cublas 12.8;
the cu121 wheel bundles cublas 12.1, which is too old.

Example for any CUDA 12.x system (including CUDA 12.2 drivers):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Step 5 — Install lammps-mdi

> **Important — numpy version:** `mace-torch` will try to pull in numpy 2.x,
> which would shadow the numpy provided by the HPC module stack (1.26.4 in
> the EasyBuild environment) and potentially break the LAMMPS Python
> interface and other compiled packages that were built against numpy 1.x.
> The `lammps-mdi` package declares `numpy<2` as a constraint to prevent
> this, but if you see numpy 2.x installed after the step below, fix it with:
> ```bash
> pip install "numpy==1.26.4"
> ```

```bash
# Recommended: with GPU-accelerated neighbor lists
pip install lammps-mdi[gpu]

# Or with cuEquivariance acceleration (base packages only):
pip install lammps-mdi[gpu-full]

# Then add the ops kernel (requires cu128 torch — see torch install above):
pip install --extra-index-url https://pypi.nvidia.com/ cuequivariance-ops-torch-cu12
```

> **Note:** `cuequivariance-ops-torch-cu12` downloads libraries from
> `https://pypi.nvidia.com/`.  On systems with SSL inspection you may need
> to add `--trusted-host pypi.nvidia.com` to the command above.
>
> You may also see a message like *"Not uninstalling numpy... outside
> environment"* during installation.  This is harmless — pip is correctly
> recognising that numpy is owned by the module system and leaving it alone.

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
