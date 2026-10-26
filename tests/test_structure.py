"""Basic structure and import tests for WATTWISE."""

import importlib
import pytest


def test_modules_importable():
    """Verify all core WATTWISE modules exist and can be imported."""
    modules = [
        "src.data_loader",
        "src.features",
        "src.models",
        "src.insights",
    ]
    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        assert mod is not None


def test_stubs_raise_not_implemented():
    """Verify that future phase stub functions raise NotImplementedError as expected before implementation."""
    from src.features import create_calendar_features
    import pandas as pd
    with pytest.raises(NotImplementedError):
        create_calendar_features(pd.DataFrame())
