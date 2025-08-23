# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

from pathlib import Path


class Config:
    def __init__(self):
        # Root of the project
        self.project_root = Path(__file__).resolve().parent

        # Data and results directories
        self.data_dir = self.project_root / "data"
        self.export_dir = self.project_root / "export"
        self.source_dir = self.project_root / "namematching"
        self.scraping_dir = self.source_dir / "scraping"
        self.databuild_dir = self.source_dir / "databuild"
        self.metrics_dir = self.source_dir / "metrics"
        self.model_dir = self.source_dir / "model"


CONFIG = Config()
