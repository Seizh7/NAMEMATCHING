# Copyright (c) 2025 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)

"""Resource management for namematching package."""

import os
from pathlib import Path
import pkg_resources


def get_resource_path(resource_name):
    """
    Get the path to a resource file (model, scaler, etc.).
    
    First tries to find the resource in the package data,
    then falls back to the export directory for development.
    
    Args:
        resource_name (str): Name of the resource file
        
    Returns:
        Path: Path to the resource file
        
    Raises:
        FileNotFoundError: If the resource cannot be found
    """
    # Try to get from package data first (when installed)
    try:
        # This works when the package is installed
        resource_path = pkg_resources.resource_filename(
            'namematching', f'data/{resource_name}'
        )
        if os.path.exists(resource_path):
            return Path(resource_path)
    except Exception:
        pass
    
    # Fall back to development structure
    package_root = Path(__file__).parent.parent
    export_dir = package_root / "export"
    resource_path = export_dir / resource_name
    
    if resource_path.exists():
        return resource_path
    
    # Try alternative locations
    alternatives = [
        package_root / "namematching" / "data" / resource_name,
        Path.cwd() / "export" / resource_name,
        Path.cwd() / "namematching" / "data" / resource_name,
    ]
    
    for alt_path in alternatives:
        if alt_path.exists():
            return alt_path
    
    raise FileNotFoundError(
        f"Could not find resource '{resource_name}'. "
        f"Searched in: {[str(p) for p in [resource_path] + alternatives]}"
    )


def get_model_path():
    """Get path to the trained model."""
    return get_resource_path("namematching_model.keras")


def get_scaler_path():
    """Get path to the trained scaler."""
    return get_resource_path("scaler.pkl")


def get_char_tokenizer_path():
    """Get path to the character tokenizer."""
    return get_resource_path("char_tokenizer.pkl")
