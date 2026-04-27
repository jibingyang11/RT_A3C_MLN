from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw' / 'swat'
PROCESSED_DATA_DIR = DATA_DIR / 'processed' / 'swat'
STREAM_DATA_DIR = DATA_DIR / 'stream' / 'swat'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
FIGURE_DIR = OUTPUT_DIR / 'figures'
LOG_DIR = OUTPUT_DIR / 'logs'
MODEL_DIR = OUTPUT_DIR / 'models'


@dataclass(frozen=True)
class WindowConfig:
    window_size: int = 2000
    step_size: int = 1000
    lag: int = 2
    min_support: float = 0.1
    min_confidence: float = 0.6
    top_attack_rules: int = 10
    top_normal_rules: int = 10


DEFAULT_WINDOW_CONFIG = WindowConfig()
