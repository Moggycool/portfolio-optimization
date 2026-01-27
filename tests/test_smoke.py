"""Smoke tests for the repository structure and basic imports."""
from pathlib import Path
import importlib


def test_repo_has_expected_folders():
    """Smoke test to ensure expected folders exist in the repo."""
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "src").exists()
    assert (repo_root / "scripts").exists()
    assert (repo_root / "outputs").exists()


def test_config_imports():
    """Smoke test to ensure config module imports without error."""
    mod = importlib.import_module("src.config")
    assert mod is not None
