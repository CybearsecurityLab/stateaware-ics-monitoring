"""Data model and exception types for st_gen."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Tank:
    id: str
    min_level: float
    max_level: float


@dataclass(frozen=True)
class Pump:
    id: str
    node1: str
    node2: str
    initial_status: str = "OPEN"


@dataclass(frozen=True)
class Valve:
    id: str
    node1: str
    node2: str
    vtype: str
    setting: float
    initial_status: str = "OPEN"


@dataclass(frozen=True)
class Control:
    link_id: str
    target_state: str           # "OPEN" | "CLOSED"
    sensor_node: str
    comparator: str             # "BELOW" | "ABOVE"
    threshold: float


@dataclass(frozen=True)
class Network:
    tanks: list[Tank]
    pumps: list[Pump]
    valves: list[Valve]
    controls: list[Control]


@dataclass(frozen=True)
class Plc:
    name: str
    sensors: list[str]
    actuators: list[str]


class StGenError(Exception):
    """Base class for all st_gen errors."""


class UnsupportedControlError(StGenError):
    """Raised when an INP [CONTROLS] line uses a form st_gen does not handle."""


class MalformedInpError(StGenError):
    """Raised when a structural INP line ([TANKS]/[PUMPS]/[VALVES]) is malformed."""


class UnsupportedRulesError(StGenError):
    """Raised when INP [RULES] section is non-empty."""


class MultiStateActuatorError(StGenError):
    """Raised when an actuator has an ambiguous (comparator, threshold) pair that maps
    to both OPEN and CLOSED in [CONTROLS]."""


class ValidationError(StGenError):
    """Raised when plcs.yaml fails schema validation."""


@dataclass(frozen=True)
class Manifest:
    topology: str
    inp: str
    plcs_yaml: str
    generated_at: str
    plcs: list[dict]
    #: Surrogate provenance block (see :mod:`st_gen.provenance`).
    provenance: dict = field(default_factory=dict)
