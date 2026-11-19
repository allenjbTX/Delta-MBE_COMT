"""Lazy reader for the aggregate MBE(3) QM/MM NPZ dataset."""

from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
from torch.utils.data import Dataset


_ELEMENTS = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn "
    "Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd "
    "In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
    "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am "
    "Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
_ATOMIC_NUMBER = {symbol: number for number, symbol in enumerate(_ELEMENTS)}

BOHR_PER_ANGSTROM = 1.8897261254578281


def _scalar_string(value: np.ndarray) -> str:
    item = np.asarray(value).item()
    if isinstance(item, bytes):
        return item.decode()
    return str(item)


def _atomic_numbers(elements: np.ndarray) -> np.ndarray:
    result = np.empty(len(elements), dtype=np.int64)
    for index, raw_symbol in enumerate(elements):
        symbol = raw_symbol.decode() if isinstance(raw_symbol, bytes) else str(raw_symbol)
        try:
            result[index] = _ATOMIC_NUMBER[symbol]
        except KeyError as exc:
            raise ValueError("Unknown element symbol {!r}".format(symbol)) from exc
    return result


def read_split_files(dataset_root: Path, split_dir: Path, split: str) -> List[Path]:
    """Resolve entries such as ``npz_files_mbe3/config_0000_mbe3.npz``."""
    manifest = split_dir / "{}_files.txt".format(split)
    if not manifest.is_file():
        raise FileNotFoundError("Missing split manifest: {}".format(manifest))
    paths = []
    for line in manifest.read_text().splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        path = Path(entry)
        if not path.is_absolute():
            path = dataset_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError("Split entry does not exist: {}".format(path))
        paths.append(path)
    if not paths:
        raise ValueError("No files listed in {}".format(manifest))
    return paths


class MBE3Dataset(Dataset):
    """Load only the compact global-system fields needed during training.

    Coordinates are returned in Angstrom and energies in Hartree.  Stored
    forces are Hartree/Bohr, so they are converted to Hartree/Angstrom to
    match derivatives with respect to the coordinate tensors.
    """

    def __init__(self, files: List[Path], validate_first: bool = True):
        self.files = list(files)
        if validate_first:
            self._validate_file(self.files[0])

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _validate_file(path: Path) -> None:
        required = {
            "qm_coords",
            "qm_elems",
            "mm_coords",
            "mm_base_charges",
            "delta_mbe_e_embedded",
            "delta_mbe_f_embedded_qm",
            "delta_mbe_f_embedded_mm",
        }
        with np.load(str(path), allow_pickle=False) as data:
            missing = required.difference(data.files)
            if missing:
                raise KeyError("{} is missing {}".format(path, sorted(missing)))
            if int(np.asarray(data["mbe_order"]).item()) != 3:
                raise ValueError("{} is not an MBE(3) aggregate".format(path))
            coordinate_unit = _scalar_string(data["coordinate_unit"]).lower()
            energy_unit = _scalar_string(data["energy_unit"]).lower()
            force_unit = _scalar_string(data["force_unit"]).lower()
            if coordinate_unit not in {"angstrom", "ang", "a"}:
                raise ValueError("Unsupported coordinate unit: {}".format(coordinate_unit))
            if energy_unit not in {"hartree", "ha"}:
                raise ValueError("Unsupported energy unit: {}".format(energy_unit))
            if force_unit not in {"hartree/bohr", "ha/bohr"}:
                raise ValueError("Unsupported force unit: {}".format(force_unit))

    def __getitem__(self, index: int) -> Dict[str, Union[str, torch.Tensor]]:
        path = self.files[index]
        with np.load(str(path), allow_pickle=False) as data:
            qm_pos = np.asarray(data["qm_coords"], dtype=np.float64).copy()
            qm_z = _atomic_numbers(np.asarray(data["qm_elems"]))
            mm_pos = np.asarray(data["mm_coords"], dtype=np.float64).copy()
            mm_charge = np.asarray(data["mm_base_charges"], dtype=np.float64).copy()
            energy = float(np.asarray(data["delta_mbe_e_embedded"]).item())
            qm_force = (
                np.asarray(data["delta_mbe_f_embedded_qm"], dtype=np.float64).copy()
                * BOHR_PER_ANGSTROM
            )
            mm_force = (
                np.asarray(data["delta_mbe_f_embedded_mm"], dtype=np.float64).copy()
                * BOHR_PER_ANGSTROM
            )
            snapshot_id = _scalar_string(data["snapshot_id"])

        if qm_force.shape != qm_pos.shape:
            raise ValueError("QM force/coordinate shape mismatch in {}".format(path))
        if mm_force.shape != mm_pos.shape or len(mm_charge) != len(mm_pos):
            raise ValueError("MM array shape mismatch in {}".format(path))

        return {
            "path": str(path),
            "snapshot_id": snapshot_id,
            "qm_z": torch.from_numpy(qm_z),
            "qm_pos": torch.from_numpy(qm_pos),
            "mm_pos": torch.from_numpy(mm_pos),
            "mm_charge": torch.from_numpy(mm_charge),
            "energy": torch.tensor(energy, dtype=torch.float64),
            "qm_force": torch.from_numpy(qm_force),
            "mm_force": torch.from_numpy(mm_force),
        }

    @staticmethod
    def energy_mean(files: List[Path]) -> float:
        total = 0.0
        for path in files:
            with np.load(str(path), allow_pickle=False) as data:
                total += float(np.asarray(data["delta_mbe_e_embedded"]).item())
        return total / len(files)
