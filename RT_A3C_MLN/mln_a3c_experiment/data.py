from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile
import urllib.request

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    urls: tuple[str, ...]
    files: tuple[str, ...]


DATASETS = {
    "mushroom": DatasetSpec(
        name="mushroom",
        urls=("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data",),
        files=("agaricus-lepiota.data",),
    ),
    "adult": DatasetSpec(
        name="adult",
        urls=(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        ),
        files=("adult.data", "adult.test"),
    ),
    "bank": DatasetSpec(
        name="bank",
        urls=("https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip",),
        files=("bank.zip",),
    ),
    "spambase": DatasetSpec(
        name="spambase",
        urls=("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",),
        files=("spambase.data",),
    ),
}


def _download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name}")
        return
    print(f"[download] {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        dest.write_bytes(response.read())
    print(f"[saved] {dest}")


def download_all(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for spec in DATASETS.values():
        dataset_dir = raw_dir / spec.name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for url, filename in zip(spec.urls, spec.files):
            _download_url(url, dataset_dir / filename)


def _ensure_downloaded(name: str, raw_dir: Path) -> None:
    spec = DATASETS[name]
    missing = [filename for filename in spec.files if not (raw_dir / name / filename).exists()]
    if missing:
        for url, filename in zip(spec.urls, spec.files):
            if filename in missing:
                _download_url(url, raw_dir / name / filename)


def _sample(frame: pd.DataFrame, max_samples: int | None, seed: int = 7) -> pd.DataFrame:
    if max_samples is None or len(frame) <= max_samples:
        return frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(frame), size=max_samples, replace=False)
    return frame.iloc[np.sort(indices)].reset_index(drop=True)


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(col).strip().replace("-", "_").replace(" ", "_") for col in frame.columns]
    for col in frame.columns:
        if frame[col].dtype == object:
            frame[col] = (
                frame[col]
                .astype(str)
                .str.strip()
                .str.replace(".", "", regex=False)
                .replace({"?": "missing", "": "missing", "nan": "missing"})
            )
    return frame


def load_dataset(name: str, raw_dir: Path, max_samples: int | None = None) -> tuple[pd.DataFrame, str]:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset: {name}")
    _ensure_downloaded(name, raw_dir)
    if name == "mushroom":
        return _load_mushroom(raw_dir / name, max_samples)
    if name == "adult":
        return _load_adult(raw_dir / name, max_samples)
    if name == "bank":
        return _load_bank(raw_dir / name, max_samples)
    if name == "spambase":
        return _load_spambase(raw_dir / name, max_samples)
    raise AssertionError(name)


def _load_mushroom(dataset_dir: Path, max_samples: int | None) -> tuple[pd.DataFrame, str]:
    columns = [
        "class",
        "cap_shape",
        "cap_surface",
        "cap_color",
        "bruises",
        "odor",
        "gill_attachment",
        "gill_spacing",
        "gill_size",
        "gill_color",
        "stalk_shape",
        "stalk_root",
        "stalk_surface_above_ring",
        "stalk_surface_below_ring",
        "stalk_color_above_ring",
        "stalk_color_below_ring",
        "veil_type",
        "veil_color",
        "ring_number",
        "ring_type",
        "spore_print_color",
        "population",
        "habitat",
    ]
    frame = pd.read_csv(dataset_dir / "agaricus-lepiota.data", header=None, names=columns)
    frame = _clean_frame(frame)
    frame["target"] = (frame["class"] == "p").astype(int)
    frame = frame.drop(columns=["class"])
    return _sample(frame, max_samples), "target"


def _load_adult(dataset_dir: Path, max_samples: int | None) -> tuple[pd.DataFrame, str]:
    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income",
    ]
    train = pd.read_csv(dataset_dir / "adult.data", header=None, names=columns, skipinitialspace=True)
    test = pd.read_csv(
        dataset_dir / "adult.test",
        header=None,
        names=columns,
        skipinitialspace=True,
        comment="|",
    )
    frame = pd.concat([train, test], ignore_index=True)
    frame = frame.dropna(how="any")
    frame = _clean_frame(frame)
    frame["target"] = frame["income"].str.contains(">50K", regex=False).astype(int)
    frame = frame.drop(columns=["income"])
    return _sample(frame, max_samples), "target"


def _load_bank(dataset_dir: Path, max_samples: int | None) -> tuple[pd.DataFrame, str]:
    zip_path = dataset_dir / "bank.zip"
    with ZipFile(zip_path) as archive:
        candidates = [name for name in archive.namelist() if name.endswith("bank-full.csv")]
        if not candidates:
            candidates = [name for name in archive.namelist() if name.endswith(".csv")]
        with archive.open(candidates[0]) as handle:
            frame = pd.read_csv(handle, sep=";")
    frame = _clean_frame(frame)
    frame["target"] = (frame["y"] == "yes").astype(int)
    frame = frame.drop(columns=["y"])
    return _sample(frame, max_samples), "target"


def _load_spambase(dataset_dir: Path, max_samples: int | None) -> tuple[pd.DataFrame, str]:
    columns = [f"feature_{i:02d}" for i in range(57)] + ["target"]
    frame = pd.read_csv(dataset_dir / "spambase.data", header=None, names=columns)
    frame = _clean_frame(frame)
    frame["target"] = frame["target"].astype(int)
    return _sample(frame, max_samples), "target"
