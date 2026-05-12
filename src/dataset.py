import random
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from src import config

@dataclass(frozen=True)
class DatasetSplit:
    train_indices: list[int]
    val_indices: list[int]
    train_patients: set[str]
    val_patients: set[str]

def discover_window_files(data_dir: str | Path) -> list[str]:
    """поиск всех предобработанных окон в формате .npz."""
    files = sorted(str(path) for path in Path(data_dir).rglob("*.npz"))
    if not files:
        print(f'No .npz windows found in {data_dir}.')
    return files

def describe_dataset(files: list[str]) -> None:
    """вывод краткой информации о датасете."""
    with np.load(files[0]) as sample:
        patch_shape = sample["patches"].shape
        ppg_shape = sample["ppg"].shape
    patients = {get_patient_id(file) for file in files}
    print(f"windows: {len(files)}")
    print(f"patients: {len(patients)}")
    print(f"sample patch shape: {patch_shape}")
    print(f"sample ppg shape: {ppg_shape}")

def split_by_patient(files: list[str], val_split: float, seed: int) -> DatasetSplit:
    """разделение окон по пациентам без пересечения train и val."""

    # собираем индексы всех окон для каждого пациента
    patient_to_indices: dict[str, list[int]] = {}
    for index, file in enumerate(files):
        patient_to_indices.setdefault(get_patient_id(file), []).append(index)

    # перемешиваем пациентов, а не отдельные окна
    patient_ids = sorted(patient_to_indices)
    random.Random(seed).shuffle(patient_ids)

    # оставляем минимум одного пациента в каждой части выборки
    val_count = int(len(patient_ids) * val_split)
    val_count = min(max(1, val_count), len(patient_ids) - 1)

    val_patients = set(patient_ids[:val_count])
    train_patients = set(patient_ids[val_count:])
    return DatasetSplit(
        train_indices = indices_for_patients(patient_to_indices, train_patients),
        val_indices = indices_for_patients(patient_to_indices, val_patients),
        train_patients = train_patients,
        val_patients = val_patients,
    )

def indices_for_patients(patient_to_indices: dict[str, list[int]], patients: set[str]) -> list[int]:
    """получение отсортированных индексов окон для выбранных пациентов."""
    return sorted(index for patient_id in patients for index in patient_to_indices[patient_id])

def get_patient_id(file: str) -> str:
    """извлечение id пациента из имени файла."""
    return Path(file).stem.split("_", 1)[0]

def build_dataloaders(dataset: Dataset, split: DatasetSplit, train_config: dict) -> tuple[DataLoader, DataLoader]:
    """создание загрузчиков данных для обучения и валидации."""
    num_workers = train_config["NUM_WORKERS"]
    common = dict(
        batch_size=train_config["BATCH_SIZE"],
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(Subset(dataset, split.train_indices), shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(Subset(dataset, split.val_indices), shuffle=False, drop_last=False, **common)
    return train_loader, val_loader

class RPPGDataset(Dataset):
    """датасет предобработанных rppg-окон, сохраненных в .npz файлах."""
    def __init__(self, files: list[str], use_frame_diff: bool = False, eps: float = 1e-6):
        self.files = files
        self.use_frame_diff = use_frame_diff
        self.eps = eps

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with np.load(self.files[index]) as sample:
            patches_np = sample["patches"]
            ppg_np = sample["ppg"]
        patches = torch.from_numpy(patches_np).float().permute(0, 1, 4, 2, 3).contiguous()
        ppg = torch.from_numpy(ppg_np).float()
        if self.use_frame_diff:
            patches = self.apply_frame_diff(patches)
        return patches, ppg

    def apply_frame_diff(self, patches: torch.Tensor) -> torch.Tensor:
        """нормализованная разность между соседними кадрами."""
        diff = torch.zeros_like(patches)
        current = patches[1:]
        previous = patches[:-1]
        numerator = current - previous
        denominator = current.abs() + previous.abs() + self.eps
        diff[1:] = numerator / denominator
        return diff
