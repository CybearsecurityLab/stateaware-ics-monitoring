"""hydraulics — typed water-network contract, templates and residual maths.

The package is split in three layers:

* :mod:`hydraulics.model` — the typed, unit-explicit network contract.
* :mod:`hydraulics.loader` — INP (via WNTR) to contract.
* :mod:`hydraulics.templates` — the five hydraulic template families and
  their deterministic compilation.

The numerical residuals themselves live in :mod:`stl.residuals` (they are
consumed by the Stage-3 STL monitor and are independent of ``rtamt``).
"""

from .loader import load_network
from .model import (
    CHANNEL_KINDS,
    ChannelSpec,
    Control,
    DemandOptions,
    DemandTerm,
    HydraulicNetwork,
    HydraulicsError,
    InpParseError,
    Junction,
    Pattern,
    Pipe,
    Provenance,
    Pump,
    PumpCurve,
    Reservoir,
    Tank,
    TimeOptions,
    UnsupportedNetworkError,
    Valve,
)
from .templates import (
    CompiledTemplate,
    TemplateFamily,
    TemplateStatus,
    compile_templates,
    missing_channels,
)

__all__ = [
    "CHANNEL_KINDS",
    "ChannelSpec",
    "CompiledTemplate",
    "Control",
    "DemandOptions",
    "DemandTerm",
    "HydraulicNetwork",
    "HydraulicsError",
    "InpParseError",
    "Junction",
    "Pattern",
    "Pipe",
    "Provenance",
    "Pump",
    "PumpCurve",
    "Reservoir",
    "Tank",
    "TemplateFamily",
    "TemplateStatus",
    "TimeOptions",
    "UnsupportedNetworkError",
    "Valve",
    "compile_templates",
    "load_network",
    "missing_channels",
]
