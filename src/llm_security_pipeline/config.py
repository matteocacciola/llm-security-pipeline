"""
config.py
The pipeline's settings as data, separated from the objects it is built with.

`SecurityPipeline.__init__` takes some thirty keyword arguments, and they
are two different kinds of thing wearing the same syntax. Some are
*settings* — a threshold, a path, a flag, a mode — that could come from a
YAML file or an environment and are the same in every process. Others are
*objects* — a Redis client, a process pool, a metrics sink, an audit logger
— that are constructed per process and cannot be written down at all.

Mixing them means a deployment's security posture is spread through a
constructor call in application code, where it cannot be diffed, reviewed
or loaded from the same place as the rest of the deployment's config. This
module pulls the settings out into `PipelineConfig`, grouped by what they
govern, serializable both ways, and validated at construction rather than
on the first request that trips over them.

    config = PipelineConfig.from_dict(yaml.safe_load(open("guard.yaml")))
    pipeline = SecurityPipeline.from_config(
        config,
        state_backend=backend,          # objects still come from code
        audit_logger=RedisStreamAuditLogger(redis),
        metrics=PrometheusMetricsSink(),
    )

Nothing that identifies a secret belongs here: the signing key is an
object argument (`scope_secret_key`/`scope_keyring`) on purpose, so a
config file that gets committed does not become a key that gets committed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any

from .resilience import FailurePolicy
from .services.rate_limiter import SessionLimits

ENFORCE = "enforce"
SHADOW = "shadow"
_ENFORCEMENT_MODES = (ENFORCE, SHADOW)
_IDENTITY_MODES = ("untrusted", "authenticated")


@dataclass(frozen=True)
class ThresholdConfig:
    """The two numbers this README repeatedly calls a product decision."""

    input_risk: float = 0.6
    output_overlap: float = 0.35

    def __post_init__(self) -> None:
        for name in ("input_risk", "output_overlap"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"ThresholdConfig.{name} must be within 0..1, got {value}.")


@dataclass(frozen=True)
class PatternConfig:
    """Where the lexical patterns come from."""

    pii_config_path: str | None = None
    include_default_pii_patterns: bool = True
    injection_config_path: str | None = None
    include_default_injection_patterns: bool = True

    def __post_init__(self) -> None:
        if self.pii_config_path is None and not self.include_default_pii_patterns:
            raise ValueError("PatternConfig: no PII patterns at all (no path and defaults off).")
        if self.injection_config_path is None and not self.include_default_injection_patterns:
            raise ValueError(
                "PatternConfig: no injection patterns at all (no path and defaults off)."
            )


@dataclass(frozen=True)
class ExfilConfig:
    allowed_hosts: tuple[str, ...] = ()
    scan_output: bool = True
    scan_tool_call_arguments: bool = True
    scan_tool_results: bool = True


@dataclass(frozen=True)
class ParallelismConfig:
    """When work is offloaded to the process pool."""

    external_scan_parallel_min_chunks: int = 2
    large_input_offload_threshold_chars: int = 20_000

    def __post_init__(self) -> None:
        if self.external_scan_parallel_min_chunks < 1:
            raise ValueError("external_scan_parallel_min_chunks must be at least 1.")
        if self.large_input_offload_threshold_chars < 0:
            raise ValueError("large_input_offload_threshold_chars cannot be negative.")


@dataclass(frozen=True)
class PipelineConfig:
    """Everything about a SecurityPipeline that is a setting, not an object.

    `session_identity` has no default here, as it has none on the
    pipeline: it is the one setting the library refuses to guess, because
    guessing "authenticated" wrongly binds session state to a claim nobody
    verified. See docs/identity-and-tokens.md, *Identity*.
    """

    session_identity: str
    enforcement: str = ENFORCE
    system_prompt: str | None = None
    # A canary is planted from a fresh random token per process, so it is a
    # flag here rather than a value: the token itself must not be in a
    # config file, and it does not need to be shared across processes —
    # each process watches for the one it planted.
    canary: bool = False
    # Which service this deployment is, for capability tokens. Not a
    # secret, so it belongs here; the key does not. See ScopeGuard.
    audience: str | None = None
    # How often a keyring provider is re-read, when one is given.
    key_reload_seconds: float = 60.0
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    patterns: PatternConfig = field(default_factory=PatternConfig)
    exfil: ExfilConfig = field(default_factory=ExfilConfig)
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    session_limits: SessionLimits = field(default_factory=SessionLimits)
    failure_policy: FailurePolicy = field(default_factory=FailurePolicy)

    def __post_init__(self) -> None:
        if self.enforcement not in _ENFORCEMENT_MODES:
            raise ValueError(
                f"PipelineConfig.enforcement must be one of {_ENFORCEMENT_MODES}, "
                f"got {self.enforcement!r}."
            )
        if self.session_identity not in _IDENTITY_MODES:
            raise ValueError(
                f"PipelineConfig.session_identity must be one of {_IDENTITY_MODES}, "
                f"got {self.session_identity!r}."
            )
        if self.key_reload_seconds <= 0:
            raise ValueError("PipelineConfig.key_reload_seconds must be positive.")
        if self.canary and self.system_prompt is None:
            raise ValueError("PipelineConfig: a canary needs a system_prompt to be planted in.")

    # -- (de)serialization ----------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Plain dicts and lists; safe to dump as JSON or YAML."""
        data = asdict(self)
        data["exfil"]["allowed_hosts"] = list(self.exfil.allowed_hosts)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        """Inverse of to_dict. Unknown keys are an error, not ignored: a
        misspelled setting that silently falls back to its default is a
        security posture that is not the one written down."""
        return _build(cls, data, path="PipelineConfig")

    # -- the bridge to the constructor ---------------------------------

    def to_kwargs(self) -> dict[str, Any]:
        """The constructor keyword arguments this config stands for."""
        return {
            "session_identity": self.session_identity,
            "enforcement": self.enforcement,
            "system_prompt": self.system_prompt,
            "canary": self.canary or None,
            "scope_audience": self.audience,
            "scope_key_reload_seconds": self.key_reload_seconds,
            "input_risk_threshold": self.thresholds.input_risk,
            "output_overlap_threshold": self.thresholds.output_overlap,
            "pii_config_path": self.patterns.pii_config_path,
            "include_default_pii_patterns": self.patterns.include_default_pii_patterns,
            "injection_config_path": self.patterns.injection_config_path,
            "include_default_injection_patterns": self.patterns.include_default_injection_patterns,
            "exfil_allowed_hosts": list(self.exfil.allowed_hosts) or None,
            "scan_output_for_exfil": self.exfil.scan_output,
            "scan_tool_call_arguments": self.exfil.scan_tool_call_arguments,
            "scan_tool_results": self.exfil.scan_tool_results,
            "external_scan_parallel_min_chunks": self.parallelism.external_scan_parallel_min_chunks,
            "large_input_offload_threshold_chars": self.parallelism.large_input_offload_threshold_chars,
            "session_limits": self.session_limits,
            "failure_policy": self.failure_policy,
        }


# Names of constructor arguments that are settings, i.e. covered by the
# config. Anything else is an object and is passed through from_config().
SETTING_KWARGS = frozenset(PipelineConfig(session_identity="untrusted").to_kwargs())


def _build(cls: type, data: Any, path: str) -> Any:
    """Recursively construct a dataclass from a dict, refusing unknown keys."""
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a mapping, got {type(data).__name__}.")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(
            f"{path}: unknown key(s) {', '.join(sorted(unknown))}. "
            f"Known: {', '.join(sorted(known))}."
        )
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        target = known[name].type
        # Nested dataclasses are resolved by name from this module and the
        # two it imports; `from __future__ import annotations` makes the
        # field types strings.
        nested = _NESTED.get(str(target).strip("'\""))
        if nested is not None and value is not None:
            kwargs[name] = _build(nested, value, f"{path}.{name}")
        elif name == "allowed_hosts" and isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


_NESTED: dict[str, type] = {
    "ThresholdConfig": ThresholdConfig,
    "PatternConfig": PatternConfig,
    "ExfilConfig": ExfilConfig,
    "ParallelismConfig": ParallelismConfig,
    "SessionLimits": SessionLimits,
    "FailurePolicy": FailurePolicy,
}

__all__ = [
    "PipelineConfig",
    "ThresholdConfig",
    "PatternConfig",
    "ExfilConfig",
    "ParallelismConfig",
    "SETTING_KWARGS",
    "ENFORCE",
    "SHADOW",
]

# Every nested type must be a dataclass, or from_dict cannot rebuild it.
# Checked at import so a future addition fails here, not in a deployment.
for _name, _type in _NESTED.items():
    if not is_dataclass(_type):
        raise TypeError(f"config: nested type {_name} is not a dataclass.")
