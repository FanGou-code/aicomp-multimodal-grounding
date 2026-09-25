"""Shared configuration for local runs and run-metadata recording.

Model-specific identity (model ids, revisions, pixel budgets) lives in
``aicomp_grounding.serving.models`` adapters; this module keeps only cross-model
constants and the YAML run-configuration loader.
"""

from __future__ import annotations

import platform
from importlib import metadata as importlib_metadata
from pathlib import Path

INFERENCE_COMPUTE_DTYPE = "bfloat16"
RUNTIME_PYTHON_VERSION = platform.python_version()

# Generic inference pixel-budget defaults recorded in run metadata (Qwen
# processor units, 28x28 per patch). Model adapters may override via kwargs.
INFERENCE_DEFAULT_MIN_PIXELS = 256 * 28 * 28
INFERENCE_DEFAULT_MAX_PIXELS = 3072 * 28 * 28

# Dependency declarations live in pyproject.toml (single source). This tuple is
# only the fallback name/version set recorded into run metadata; the four model
# libraries are kept identical to pyproject by tests/test_env_contract.py.
# current_runtime_packages() overwrites each entry with the actually-installed
# version, so the pinned values below matter only when a package is absent.
RUNTIME_PACKAGES = (
    "transformers==5.17.0",
    "accelerate==1.15.0",
    "peft==0.21.0",
    "qwen-vl-utils==0.0.14",
    "flash-linear-attention==0.5.2",
    "pillow==12.1.0",
    "pyyaml==6.0.2",
    "torch==2.14.0",
    "torchvision==0.29.0",
)


def current_runtime_packages() -> list[str]:
    """Return installed runtime versions, with pinned fallbacks."""
    packages = list(RUNTIME_PACKAGES)

    for index, package in enumerate(packages):
        distribution, separator, _ = package.partition("==")
        if not separator:
            continue
        try:
            version = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            continue
        packages[index] = f"{distribution}=={version}"

    try:
        import torch
    except Exception:
        torch = None
    if torch is not None:
        packages = [
            f"torch=={torch.__version__}" if package.startswith("torch==") else package
            for package in packages
        ]

    return packages

CHECKPOINT_VERSION = 5
ANNOTATION_PROTOCOL_VERSION = 12
PREPARATION_PROTOCOL_VERSION = 2
TRAINING_PROTOCOL_VERSION = 2

INFERENCE_SPLITS = frozenset({"train", "val", "test"})


def load_run_config(path: str | Path) -> dict[str, dict]:
    """Load a YAML run configuration with ``run`` / ``hyperparameters`` sections.

    Unknown sections are rejected so a typo cannot silently create a section
    nothing reads.  Key validation against each entrypoint's accepted
    parameters happens in :func:`merge_run_config`.
    """
    import yaml

    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Run config must be a YAML mapping: {config_path}")
    unknown = sorted(set(raw) - {"run", "hyperparameters"})
    if unknown:
        raise ValueError(f"Run config has unknown sections: {unknown}")
    sections: dict[str, dict] = {}
    for name in ("run", "hyperparameters"):
        value = raw.get(name) or {}
        if not isinstance(value, dict):
            raise ValueError(f"Run config section {name!r} must be a mapping")
        sections[name] = value
    return sections


def merge_run_config(
    args,
    config: dict[str, dict],
    *,
    parser,
    run_keys: tuple[str, ...],
    hyperparameter_keys: tuple[str, ...],
) -> None:
    """Fill CLI-unspecified arguments from a run config; CLI wins over YAML.

    ``args`` is mutated in place.  A parameter counts as unspecified when its
    value is ``None``; the entrypoint applies its real defaults after this
    call.  Keys outside ``run_keys`` / ``hyperparameter_keys`` are rejected so
    a typo cannot silently fall back to a default.  String scalars pass through
    the matching argparse ``type`` so ``data_dir: data`` and ``--data-dir data``
    resolve identically.
    """
    type_by_dest = {action.dest: action.type for action in parser._actions}
    for section, allowed in (
        ("run", run_keys),
        ("hyperparameters", hyperparameter_keys),
    ):
        unknown = sorted(set(config.get(section, {})) - set(allowed))
        if unknown:
            raise ValueError(f"Run config has unknown {section} keys: {unknown}")
        for key, value in config.get(section, {}).items():
            if getattr(args, key) is not None:
                continue
            convert = type_by_dest.get(key)
            if convert is not None and isinstance(value, str):
                value = convert(value)
            setattr(args, key, value)
