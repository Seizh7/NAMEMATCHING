# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

from setuptools import setup, find_packages

setup(
    name="namematching",
    version="1.0",
    packages=find_packages(),
    package_data={
        'namematching': ['data/*', 'model/*'],
    },
    include_package_data=True,
    install_requires=[
        # Apple Silicon (arm64 macOS) → tensorflow-macos
        "tensorflow-macos>=2.10.0; platform_machine == 'arm64' and sys_platform == 'darwin'",
        # Tout le reste (Intel Mac, Linux, Windows)
        "tensorflow>=2.10.0; platform_machine != 'arm64' or sys_platform != 'darwin'",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "textdistance>=4.2.0",
    ],
    author="Seizh7",
    description="Système de comparaison de noms basé sur l'IA",
    python_requires=">=3.8",
)
