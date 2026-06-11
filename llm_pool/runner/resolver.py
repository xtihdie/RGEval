from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Type

from ..llm import (
    DeepSeekClient,
    QwenClient,
    QwenVLClient,
    ZhipuClient,
    ZhipuVLClient,
)
from ..llm.config import get_provider_mappings


# ============================================================================
# ① 可扩展声明区（唯一需要维护的地方）
# ============================================================================

@dataclass(frozen=True)
class ProviderConfig:
    name: str
    provider: str
    aliases: Iterable[str]
    default_model: str
    version_to_model: Dict[str, str]   # semantic version -> real model
    max_workers: int
    supports_post: bool


CLIENT_FACTORY: Dict[str, Type] = {
    "deepseek": DeepSeekClient,
    "qwen": QwenClient,
    "qwen-vl": QwenVLClient,
    "zhipu": ZhipuClient,
    "zhipu-vl": ZhipuVLClient,
}


def _load_provider_configs() -> Dict[str, ProviderConfig]:
    raw = get_provider_mappings()
    if not raw:
        raise ValueError("No provider_mappings found in config.toml")

    configs: Dict[str, ProviderConfig] = {}
    for name, cfg in raw.items():
        provider_type = cfg.get("provider") or name
        aliases = cfg.get("aliases", [])
        default_model = cfg.get("default_model")
        versions = cfg.get("versions", {})
        max_workers = int(cfg.get("max_workers", 4))
        supports_post = bool(cfg.get("supports_post", True))

        if provider_type not in CLIENT_FACTORY:
            raise ValueError(
                f"Unknown provider type {provider_type!r} for mapping {name!r}"
            )
        if not default_model:
            raise ValueError(f"Missing default_model for provider: {name}")
        if not aliases:
            raise ValueError(f"Missing aliases for provider: {name}")
        if not versions:
            raise ValueError(f"Missing versions for provider: {name}")

        configs[name] = ProviderConfig(
            name=name,
            provider=provider_type,
            aliases=tuple(aliases),
            default_model=default_model,
            version_to_model=dict(versions),
            max_workers=max_workers,
            supports_post=supports_post,
        )

    return configs


_provider_configs: Dict[str, ProviderConfig] | None = None
_alias_index: Dict[str, str] | None = None


def get_provider_configs() -> Dict[str, ProviderConfig]:
    global _provider_configs
    if _provider_configs is None:
        _provider_configs = _load_provider_configs()
    return _provider_configs


# ============================================================================
# ② 自动索引
# ============================================================================

def get_alias_index() -> Dict[str, str]:
    global _alias_index
    if _alias_index is None:
        _alias_index = {
            alias.lower(): name
            for name, cfg in get_provider_configs().items()
            for alias in cfg.aliases
        }
    return _alias_index


# ============================================================================
# ③ 字符串规范化（⭐ 核心能力）
# ============================================================================

_VERSION_PATTERN = re.compile(r"(\d+(?:\.\d+)?)")


def normalize_model_input(
    raw: str,
    provider: str,
) -> str | None:
    """
    Normalize model/version input to semantic version string.

    Examples:
        "v3"            -> "3"
        "3.1"           -> "3.1"
        "deepseekv3.1"  -> "3.1"
        "deepseek_3"    -> "3"
        "DeepSeek-3.1"  -> "3.1"
        "dsv3.1"        -> "3.1"
    """
    if not raw:
        return None

    s = raw.lower()

    # remove provider name
    s = s.replace(provider, "")

    # remove common separators
    for ch in ("_", "-", "v"):
        s = s.replace(ch, "")

    # extract version number
    m = _VERSION_PATTERN.search(s)
    if not m:
        return None

    return m.group(1)


# ============================================================================
# ④ Resolver 核心逻辑
# ============================================================================

def parse_provider_arg(arg: str | None) -> Tuple[str | None, str | None]:
    if arg is None:
        return None, None
    if ":" in arg:
        p, m = arg.split(":", 1)
        return p.strip(), m.strip()
    return arg, None


def resolve_provider_and_model(
    provider: str | None = None,
    model: str | None = None,
) -> Tuple[str, str, str]:

    provider, embedded_model = parse_provider_arg(provider)
    model = model or embedded_model

    # ---------------- provider ----------------
    raw_provider = (provider or "deepseek").lower()
    provider_name = get_alias_index().get(raw_provider)

    if provider_name is None:
        raise ValueError(f"Unknown provider: {provider!r}")

    cfg = get_provider_configs()[provider_name]

    # ---------------- model ----------------
    if model is None:
        return provider_name, cfg.provider, cfg.default_model

    semantic_version = normalize_model_input(model, provider_name)

    if semantic_version is None:
        raise ValueError(
            f"Cannot parse model/version: {model!r} for provider {provider_name}"
        )

    real_model = cfg.version_to_model.get(semantic_version)
    if real_model is None:
        raise ValueError(
            f"Unsupported version {semantic_version!r} for provider {provider_name}. "
            f"Supported: {sorted(cfg.version_to_model)}"
        )

    return provider_name, cfg.provider, real_model


def resolve_max_workers(provider: str | None = None) -> int:
    raw_provider = (provider or "deepseek").lower()
    provider_name = get_alias_index().get(raw_provider)
    if provider_name is None:
        raise ValueError(f"Unknown provider: {provider!r}")
    return get_provider_configs()[provider_name].max_workers


def resolve_supports_post(provider: str | None = None) -> bool:
    raw_provider = (provider or "deepseek").lower()
    provider_name = get_alias_index().get(raw_provider)
    if provider_name is None:
        raise ValueError(f"Unknown provider: {provider!r}")
    return get_provider_configs()[provider_name].supports_post


# ============================================================================
# ⑤ Public API
# ============================================================================

def resolve_client(
    provider: str | None = None,
    model: str | None = None,
    supports_post: bool | None = None,
):
    provider_name, provider_type, model_id = resolve_provider_and_model(provider, model)
    client_cls = CLIENT_FACTORY[provider_type]
    if supports_post is None:
        supports_post = resolve_supports_post(provider)
    return client_cls(
        model=model_id,
        config_name=provider_name,
        supports_post=supports_post,
    )
