"""Guards the WNTR installation that the hydraulics tests rely on.

The other hydraulics modules skip themselves when WNTR is absent, which would
hide a broken WNTR installation. WNTR 1.5.0 imports ``six`` without declaring
it as a dependency, so this check fails loudly instead of skipping.
"""

from __future__ import annotations

import pytest


def test_wntr_is_installed_and_fully_importable():
    try:
        import wntr
    except ModuleNotFoundError as exc:
        if exc.name == "wntr":
            pytest.skip("WNTR is not installed")
        pytest.fail(
            f"WNTR is installed but its dependency {exc.name!r} is missing; "
            "install it before running the hydraulics tests",
            pytrace=False,
        )
    from wntr.network import WaterNetworkModel

    assert WaterNetworkModel is not None
