"""AIrPPG Task 4 - Lightweight model package."""

from airppg.modeling.config import ModelingConfig
from airppg.modeling.models import SmallTCN, TinyCNN1D

__all__ = ["ModelingConfig", "TinyCNN1D", "SmallTCN"]
