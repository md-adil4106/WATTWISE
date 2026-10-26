"""Integration and UI tests for WATTWISE Streamlit dashboard using AppTest."""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_loads_overview_without_exceptions():
    """Verify that WATTWISE loads successfully on the Overview page with zero exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=35)
    at.run()
    assert len(at.exception) == 0
    # Ensure title is present
    assert any("WATTWISE" in str(title.value) for title in at.title)


def test_app_all_navigation_pages_without_exceptions():
    """Verify that every tab in WATTWISE renders cleanly with zero exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=35)
    at.run()
    assert len(at.exception) == 0

    pages = ["Overview", "Forecast", "PeakWatch", "Model Arena", "Pattern Lens"]
    for page in pages:
        at.radio[0].set_value(page).run()
        assert len(at.exception) == 0, f"Exception found on page {page}: {[e.message for e in at.exception]}"


def test_app_synthetic_mode_shows_warning_banner():
    """Verify that selecting Synthetic Demo mode displays warning banner and runs cleanly."""
    at = AppTest.from_file(APP_PATH, default_timeout=35)
    at.run()
    assert len(at.exception) == 0

    # Select Synthetic Demo mode
    at.selectbox[0].set_value("Synthetic Demo").run()
    assert len(at.exception) == 0
    # Confirm warning banner exists
    assert any("SYNTHETIC DEMO" in str(w.value) for w in at.warning)
