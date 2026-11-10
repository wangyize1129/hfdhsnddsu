from dataclasses import dataclass


@dataclass
class TASKS:
    RECONSTRUCTION: str = "best"
    FORECASTING: str = "forecasting"
    CLASSIFICATION: str = "classification"
    EMBED: str = "embedding"
