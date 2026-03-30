#!/usr/bin/env python
# MolSSI lammps_step:mace_mdi 1.0
"""
Optimized MACE MDI Engine for LAMMPS.

Bypasses ASE Calculator overhead by building MACE model inputs directly.
Supports GPU-accelerated neighbor lists via vesin-torch when available,
falling back to matscipy on CPU.

Usage (direct)::

    mpirun -np 1 mace-mdi -mdi "-role ENGINE -name MACE -method MPI" \\
        : -np 1 lmp -mdi "-role DRIVER -name LAMMPS -method MPI" -in input.dat

Usage (via mdi_bind.sh for CPU/GPU pinning on standalone machines)::

    SEAMM_GPUS=0 mpirun --mca mpi_yield_when_idle 1 \\
        -np 1 mdi_bind.sh mace-mdi -mdi "-role ENGINE -name MACE -method MPI" \\
        : -np 1 mdi_bind.sh lmp -mdi "-role DRIVER -name LAMMPS -method MPI" -in input.dat

Environment variables::

    SEAMM_FF          Path to the MACE model file (.model or .pt)  [required]
    SEAMM_DEVICE      PyTorch device override (default: cuda:0)
    SEAMM_DTYPE       Model dtype: float32 or float64 (default: float32)

Authors: Paul Saxe, with assistance from Claude (Anthropic)
License: MIT
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import mdi
from mpi4py import MPI
import pint

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

# Unit conversion factors
_ureg = pint.UnitRegistry()
Bohr = _ureg.Quantity(1, "bohr").to("angstrom").magnitude  # Bohr -> Å
Hartree = _ureg.Quantity(1, "hartree").to("eV").magnitude  # Hartree -> eV


# ---------------------------------------------------------------------------
# Optional neighbor list backends
# ---------------------------------------------------------------------------

try:
    from vesin.torch import NeighborList as VesinNeighborList

    VESIN_AVAILABLE = True
except ImportError:
    VESIN_AVAILABLE = False

if not VESIN_AVAILABLE:
    from matscipy.neighbours import neighbour_list


# ---------------------------------------------------------------------------
# Optional acceleration backends
# ---------------------------------------------------------------------------

try:
    from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq

    CUEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CUEQ_AVAILABLE = False

try:
    from mace.cli.convert_e3nn_oeq import run as run_e3nn_to_oeq

    OEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    OEQ_AVAILABLE = False


# ---------------------------------------------------------------------------
# CPU neighbor list (matscipy fallback)
# ---------------------------------------------------------------------------


def get_neighborhood_cpu(
    positions: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
    pbc: tuple = (True, True, True),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute neighbor list on CPU using matscipy.

    Parameters
    ----------
    positions : np.ndarray, shape [N, 3], in Angstroms
    cell      : np.ndarray, shape [3, 3], in Angstroms
    cutoff    : float, in Angstroms
    pbc       : tuple of 3 bools

    Returns
    -------
    edge_index   : np.ndarray [2, E]
    shifts       : np.ndarray [E, 3]   real-space shift vectors (Å)
    unit_shifts  : np.ndarray [E, 3]   integer lattice shifts
    """
    cell_copy = np.array(cell, dtype=float)

    # Extend non-periodic directions so matscipy finds all neighbors
    if not all(pbc):
        identity = np.identity(3, dtype=float)
        max_pos = np.max(np.absolute(positions)) + 1
        for dim in range(3):
            if not pbc[dim]:
                cell_copy[dim, :] = max_pos * 5 * cutoff * identity[dim, :]

    sender, receiver, unit_shifts = neighbour_list(
        quantities="ijS",
        pbc=pbc,
        cell=cell_copy,
        positions=positions,
        cutoff=cutoff,
    )

    # Remove zero-shift self-edges
    true_self_edge = (sender == receiver) & np.all(unit_shifts == 0, axis=1)
    keep = ~true_self_edge
    sender, receiver, unit_shifts = sender[keep], receiver[keep], unit_shifts[keep]

    edge_index = np.stack((sender, receiver))
    shifts = np.dot(unit_shifts, cell_copy)
    return edge_index, shifts, unit_shifts


# ---------------------------------------------------------------------------
# MACEEngine
# ---------------------------------------------------------------------------


class MACEEngine:
    """MDI engine that runs a MACE model, communicating with LAMMPS via MDI.

    Bypasses the ASE Calculator interface for lower per-step overhead,
    building MACE model inputs directly from MDI data.

    Parameters
    ----------
    model_path   : str   Path to the saved MACE model (.model or .pt)
    device       : str   PyTorch device string, e.g. "cuda:0" or "cpu"
    default_dtype: str   "float32" or "float64"
    enable_cueq  : bool  Convert model to cuEquivariance (requires NVIDIA GPU)
    enable_oeq   : bool  Convert model to openEquivariance
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        default_dtype: str = "float32",
        enable_cueq: bool = False,
        enable_oeq: bool = False,
    ):
        self.device = torch.device(device)
        self.dtype = torch.float32 if default_dtype == "float32" else torch.float64

        # ---- Load model ----
        model = torch.load(f=model_path, map_location=self.device, weights_only=False)

        model_dtype = next(model.parameters()).dtype
        if model_dtype != self.dtype:
            logging.warning(f"Model dtype {model_dtype} != requested {self.dtype}, converting.")
            model = model.double() if self.dtype == torch.float64 else model.float()

        # ---- Apply acceleration ----
        if enable_cueq:
            if not CUEQ_AVAILABLE:
                raise ImportError(
                    "cuEquivariance is not installed. "
                    "pip install cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu12"
                )
            logging.info("Converting model to CuEq for acceleration")
            model = run_e3nn_to_cueq(model, device=str(self.device)).to(self.device)
        elif enable_oeq:
            if not OEQ_AVAILABLE:
                raise ImportError("openEquivariance is not installed. pip install openequivariance")
            logging.info("Converting model to OEq for acceleration")
            model = run_e3nn_to_oeq(model, device=str(self.device)).to(self.device)

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        self.model = model
        self.r_max = float(model.r_max.cpu())

        # Atomic number -> one-hot index mapping
        self.atomic_numbers = [int(z) for z in model.atomic_numbers]
        self.z_to_index = {z: i for i, z in enumerate(self.atomic_numbers)}
        self.num_species = len(self.atomic_numbers)

        try:
            self.heads = list(model.heads)
        except AttributeError:
            self.heads = ["Default"]
        self.head_index = 0

        logging.info(
            f"Model loaded: r_max={self.r_max:.3f} Å, "
            f"species={self.atomic_numbers}, "
            f"heads={self.heads}, "
            f"dtype={self.dtype}, device={self.device}"
        )
        if VESIN_AVAILABLE:
            logging.info("vesin-torch available — using GPU neighbor lists")
            self.vesin_nl = VesinNeighborList(cutoff=self.r_max, full_list=True)
        else:
            logging.info("vesin-torch not available — using matscipy CPU neighbor lists")
            self.vesin_nl = None

        # ---- MDI state ----
        self.natoms: int | None = None
        self.elements_np: np.ndarray | None = None
        self.positions_np: np.ndarray | None = None  # Bohr (MDI units)
        self.cell_np: np.ndarray | None = None  # Bohr
        self.periodic: bool = False

        # ---- Cached GPU tensors (reallocated when natoms/elements change) ----
        self._node_attrs = None
        self._batch = None
        self._ptr = None
        self._head = None
        self._num_graphs = None
        self._pbc = None

        # ---- Results ----
        self.energy: float | None = None
        self.forces: np.ndarray | None = None
        self.stress: np.ndarray | None = None
        self._needs_calculation: bool = True

        # ---- Timing ----
        self._n_calc = 0
        self._t_nlist = self._t_transfer = self._t_model = self._t_total = 0.0

    def _init_persistent_tensors(self, natoms: int, elements: np.ndarray) -> None:
        """Allocate tensors that are constant across MD steps (node attributes, batch)."""
        indices = [self.z_to_index[int(z)] for z in elements]
        one_hot = torch.zeros(natoms, self.num_species, dtype=self.dtype, device=self.device)
        for i, idx in enumerate(indices):
            one_hot[i, idx] = 1.0
        self._node_attrs = one_hot
        self._batch = torch.zeros(natoms, dtype=torch.long, device=self.device)
        self._ptr = torch.tensor([0, natoms], dtype=torch.long, device=self.device)
        self._head = torch.tensor([self.head_index], dtype=torch.long, device=self.device)
        self._num_graphs = torch.tensor(1, dtype=torch.long, device=self.device)

    def _build_graph_vesin(
        self,
        positions_t: torch.Tensor,
        cell_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build graph edges on GPU using vesin-torch."""
        i, j, S, _ = self.vesin_nl.compute(
            points=positions_t,
            box=cell_t,
            periodic=self.periodic,
            quantities="ijSd",
        )
        # Remove zero-shift self-edges
        self_edge = (i == j) & (S == 0).all(dim=1)
        keep = ~self_edge
        i, j, S = i[keep], j[keep], S[keep]
        edge_index = torch.stack([i, j], dim=0)
        shifts = S.to(dtype=self.dtype) @ cell_t
        return edge_index, shifts, S.to(dtype=self.dtype)

    def _build_graph_cpu(
        self,
        positions_np: np.ndarray,
        cell_np: np.ndarray,
        pbc: tuple = (True, True, True),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build graph edges on CPU using matscipy, then transfer to device."""
        edge_index_np, shifts_np, unit_shifts_np = get_neighborhood_cpu(
            positions_np, cell_np, self.r_max, pbc=pbc
        )
        edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=self.device)
        shifts = torch.tensor(shifts_np, dtype=self.dtype, device=self.device)
        unit_shifts = torch.tensor(unit_shifts_np, dtype=self.dtype, device=self.device)
        return edge_index, shifts, unit_shifts

    def calculate(self) -> None:
        """Run one MACE evaluation: build graph, forward pass, extract results."""
        t_start = time.perf_counter()

        # MDI Bohr -> Angstrom
        positions_ang = self.positions_np * Bohr

        if self.periodic:
            cell_ang = self.cell_np * Bohr
            pbc = (True, True, True)
            compute_stress = True
        else:
            max_pos = np.max(np.absolute(positions_ang)) + 1
            fake_size = max_pos * 5 * self.r_max
            cell_ang = np.diag([fake_size, fake_size, fake_size])
            pbc = (False, False, False)
            compute_stress = False

        if self._pbc is None:
            self._pbc = torch.tensor([list(pbc)], dtype=torch.bool, device=self.device)

        # ---- Neighbor list ----
        t0 = time.perf_counter()
        if self.vesin_nl is not None:
            positions_t = torch.tensor(positions_ang, dtype=self.dtype, device=self.device)
            cell_t = torch.tensor(cell_ang, dtype=self.dtype, device=self.device)
            edge_index, shifts, unit_shifts = self._build_graph_vesin(positions_t, cell_t)
        else:
            edge_index, shifts, unit_shifts = self._build_graph_cpu(
                positions_ang, cell_ang, pbc=pbc
            )
            positions_t = torch.tensor(positions_ang, dtype=self.dtype, device=self.device)
            cell_t = torch.tensor(cell_ang, dtype=self.dtype, device=self.device)
        t1 = time.perf_counter()

        # ---- Build input dict ----
        positions_t.requires_grad_(True)
        input_dict = {
            "positions": positions_t,
            "node_attrs": self._node_attrs,
            "edge_index": edge_index,
            "shifts": shifts,
            "unit_shifts": unit_shifts,
            "cell": cell_t.unsqueeze(0),
            "batch": self._batch,
            "ptr": self._ptr,
            "head": self._head,
            "num_graphs": self._num_graphs,
            "pbc": self._pbc,
        }
        t2 = time.perf_counter()

        # ---- Forward pass ----
        out = self.model(input_dict, compute_stress=compute_stress, training=False)
        t3 = time.perf_counter()

        # ---- Extract results, convert to MDI atomic units ----
        self.energy = out["energy"].detach().cpu().item() / Hartree
        self.forces = out["forces"].detach().cpu().to(torch.float64).numpy() / (Hartree / Bohr)
        if out.get("stress") is not None:
            self.stress = -out["stress"].detach().cpu().to(torch.float64).numpy().reshape(3, 3) / (
                Hartree / Bohr**3
            )
        else:
            self.stress = None

        t_end = time.perf_counter()

        # ---- Timing ----
        self._n_calc += 1
        self._t_nlist += t1 - t0
        self._t_transfer += t2 - t1
        self._t_model += t3 - t2
        self._t_total += t_end - t_start

        if self._n_calc % 100 == 0:
            n = self._n_calc
            logging.info(
                f"Step {n}: "
                f"nlist={self._t_nlist/n*1e3:.1f} ms  "
                f"transfer={self._t_transfer/n*1e3:.1f} ms  "
                f"model={self._t_model/n*1e3:.1f} ms  "
                f"total={self._t_total/n*1e3:.1f} ms  "
                f"rate={self.natoms*n/self._t_total/1e3:.1f} katom-step/s"
            )

    # -----------------------------------------------------------------------
    # MDI communication loop
    # -----------------------------------------------------------------------

    def run(self, mdi_args: str) -> None:
        """Main MDI engine loop.

        Parameters
        ----------
        mdi_args : str
            The MDI initialization string, e.g.
            "-role ENGINE -name MACE -method MPI"
        """
        mdi.MDI_Init(mdi_args, MPI.COMM_WORLD)

        mdi.MDI_Register_Node("@DEFAULT")
        for cmd in [
            ">NATOMS",
            ">COORDS",
            ">CELL",
            ">ELEMENTS",
            "<ENERGY",
            "<FORCES",
            "<STRESS",
            "SCF",
            "EXIT",
        ]:
            mdi.MDI_Register_Command("@DEFAULT", cmd)

        comm = mdi.MDI_Accept_Communicator()
        logging.info("MDI connection established")

        while True:
            command = mdi.MDI_Recv_Command(comm)
            logging.debug(f"MDI command: {command}")

            if command == "EXIT":
                break

            elif command == ">NATOMS":
                self.natoms = mdi.MDI_Recv(1, mdi.MDI_INT, comm)

            elif command == ">ELEMENTS":
                elements = mdi.MDI_Recv(self.natoms, mdi.MDI_INT, comm)
                self.elements_np = np.array(elements, dtype=np.int64)
                self._init_persistent_tensors(self.natoms, self.elements_np)
                logging.info(
                    f"Received {self.natoms} atoms, "
                    f"elements: {sorted(set(self.elements_np.tolist()))}"
                )

            elif command == ">CELL":
                cell = mdi.MDI_Recv(9, mdi.MDI_DOUBLE, comm)
                self.cell_np = np.array(cell, dtype=np.float64).reshape(3, 3)
                if not self.periodic:
                    self.periodic = True
                    self._pbc = torch.tensor(
                        [[True, True, True]], dtype=torch.bool, device=self.device
                    )
                    logging.info("Periodic system detected")
                self._needs_calculation = True

            elif command == ">COORDS":
                coords = mdi.MDI_Recv(3 * self.natoms, mdi.MDI_DOUBLE, comm)
                self.positions_np = np.array(coords, dtype=np.float64).reshape(self.natoms, 3)
                self._needs_calculation = True

            elif command == "<ENERGY":
                if self._needs_calculation:
                    self.calculate()
                    self._needs_calculation = False
                mdi.MDI_Send(self.energy, 1, mdi.MDI_DOUBLE, comm)

            elif command == "<FORCES":
                if self._needs_calculation:
                    self.calculate()
                    self._needs_calculation = False
                mdi.MDI_Send(self.forces.flatten(), 3 * self.natoms, mdi.MDI_DOUBLE, comm)

            elif command == "<STRESS":
                if self._needs_calculation:
                    self.calculate()
                    self._needs_calculation = False
                payload = self.stress if self.stress is not None else np.zeros(9)
                mdi.MDI_Send(payload.flatten(), 9, mdi.MDI_DOUBLE, comm)

            elif command == "SCF":
                self.calculate()
                self._needs_calculation = False

            else:
                print(f"Error: unhandled MDI command '{command}'!", file=sys.stderr)
                sys.exit(1)

        logging.info(
            f"Engine finished. {self._n_calc} calculations, "
            f"avg {self._t_total / max(self._n_calc, 1) * 1e3:.1f} ms/step"
        )

        # Clean up GPU memory before MPI tears down
        import gc

        torch.cuda.synchronize()
        del self.model, self._node_attrs
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mace-mdi",
        description="MACE MDI Engine — serves MACE force evaluations to a LAMMPS driver.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables:
  SEAMM_FF      Path to the MACE model file (.model or .pt)  [required if --model not given]
  SEAMM_DEVICE  PyTorch device override (default: cuda:0)
  SEAMM_DTYPE   Model dtype: float32 or float64 (default: float32)

Example:
  mpirun --mca mpi_yield_when_idle 1 \\
    -np 1 mdi_bind.sh mace-mdi -mdi "-role ENGINE -name MACE -method MPI" \\
    : -np 1 mdi_bind.sh lmp -mdi "-role DRIVER -name LAMMPS -method MPI" -in input.dat
""",
    )
    parser.add_argument(
        "-mdi",
        "--mdi",
        required=True,
        dest="mdi_args",
        metavar="MDI_STRING",
        help='MDI initialization string, e.g. "-role ENGINE -name MACE -method MPI"',
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        metavar="PATH",
        help="Path to the MACE model file (overrides SEAMM_FF environment variable)",
    )
    parser.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="PyTorch device string (default: $SEAMM_DEVICE or 'cuda:0')",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["float32", "float64"],
        help="Model floating-point precision (default: $SEAMM_DTYPE or 'float32')",
    )
    parser.add_argument(
        "--enable-cueq",
        action="store_true",
        default=False,
        help="Convert model to cuEquivariance representation (requires cuequivariance-torch)",
    )
    parser.add_argument(
        "--enable-oeq",
        action="store_true",
        default=False,
        help="Convert model to openEquivariance representation (requires openequivariance)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Entry point for the ``mace-mdi`` console script."""
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve model path: CLI arg > environment variable
    model_path = args.model or os.environ.get("SEAMM_FF")
    if model_path is None:
        print(
            "Error: no model path specified. "
            "Use --model PATH or set SEAMM_FF environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    device = args.device or os.environ.get("SEAMM_DEVICE", "cuda:0")
    dtype = args.dtype or os.environ.get("SEAMM_DTYPE", "float32")

    engine = MACEEngine(
        model_path=model_path,
        device=device,
        default_dtype=dtype,
        enable_cueq=args.enable_cueq,
        enable_oeq=args.enable_oeq,
    )
    engine.run(args.mdi_args)


if __name__ == "__main__":
    main()
