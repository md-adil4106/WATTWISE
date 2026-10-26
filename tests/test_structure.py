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


def test_core_module_callables():
    """Verify that all core Phase 1-3 functions are callable across modules."""
    from src.data_loader import get_data, load_raw
    from src.features import build_features, make_row
    from src.models import chrono_split, evaluate, forecast_next_24
    from src.insights import hourly_profile, peak_summary

    assert callable(load_raw)
    assert callable(get_data)
    assert callable(build_features)
    assert callable(make_row)
    assert callable(chrono_split)
    assert callable(evaluate)
    assert callable(forecast_next_24)
    assert callable(hourly_profile)
    assert callable(peak_summary)
