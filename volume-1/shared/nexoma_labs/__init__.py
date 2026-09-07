"""Shared helpers for Applied AI Engineering laboratories.

Standard library only. Labs that need third-party packages declare them in their
own pyproject.toml; this package stays importable everywhere so that cost
accounting and fixture playback work even in a bare environment.
"""

from .costs import CostLedger, Price, estimate
from .fixtures import FixtureStore, lab_mode

__all__ = ["CostLedger", "Price", "estimate", "FixtureStore", "lab_mode"]
__version__ = "1.0.0"
