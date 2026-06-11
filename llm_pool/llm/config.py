import os
import re
from pathlib import Path
import tomllib

ENV_CONFIG_PATH = "LLM_POOL_CONFIG"
PUBLIC_CONFIG_FILENAME = "config.toml"
LOCAL_CONFIG_FILENAME = "config.local.toml"
_ENV_PLACEHOLDER_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")
_config: dict | None = None


def _candidate_config_paths(filename: str = PUBLIC_CONFIG_FILENAME) -> list[Path]:
    current = Path(__file__).resolve()
    candidates = [
        Path.cwd() / filename,
        current.parents[2] / filename,
        current.parents[3] / filename,
    ]
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def _first_existing_config_path(filename: str) -> Path | None:
    for path in _candidate_config_paths(filename):
        if path.exists():
            return path
    return None


def get_config_path() -> Path:
    env_path = Path(os.environ[ENV_CONFIG_PATH]).expanduser() if ENV_CONFIG_PATH in os.environ else None
    if env_path is not None:
        if env_path.exists():
            return env_path
        raise FileNotFoundError(f"Config not found via {ENV_CONFIG_PATH}: {env_path}")

    local_path = _first_existing_config_path(LOCAL_CONFIG_FILENAME)
    if local_path is not None:
        return local_path

    public_path = _first_existing_config_path(PUBLIC_CONFIG_FILENAME)
    if public_path is not None:
        return public_path

    searched = ", ".join(
        str(path)
        for filename in (LOCAL_CONFIG_FILENAME, PUBLIC_CONFIG_FILENAME)
        for path in _candidate_config_paths(filename)
    )
    raise FileNotFoundError(
        f"Config not found. Checked {searched}. "
        f"You can also set {ENV_CONFIG_PATH} to a TOML config path."
    )


def _load_config_file(config_path: Path) -> dict:
    with config_path.open("rb") as f:
        return tomllib.load(f)


def _deep_merge(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config() -> dict:
    if ENV_CONFIG_PATH in os.environ:
        return _load_config_file(get_config_path())

    public_path = _first_existing_config_path(PUBLIC_CONFIG_FILENAME)
    local_path = _first_existing_config_path(LOCAL_CONFIG_FILENAME)
    if public_path is None and local_path is None:
        get_config_path()

    config = _load_config_file(public_path) if public_path is not None else {}
    if local_path is not None:
        config = _deep_merge(config, _load_config_file(local_path))
    return config


def _expand_env_placeholders(value: object, config_name: str, key: str) -> object:
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        env_name = match.group(1)
        default = match.group(2)
        env_value = os.environ.get(env_name)
        if env_value is not None:
            return env_value
        if default is not None:
            return default
        raise ValueError(
            f"Missing environment variable {env_name} required by "
            f"[providers.{config_name}].{key}"
        )

    return _ENV_PLACEHOLDER_PATTERN.sub(replace, value)


def get_llm_config(name: str) -> dict:
    global _config
    if _config is None:
        _config = _load_config()
    providers = _config.get("providers", {})
    if name not in providers:
        raise ValueError(f"Config section [providers.{name}] not found in config.toml")

    provider = providers[name]
    return {
        "api_key": _expand_env_placeholders(provider.get("api_key"), name, "api_key"),
        "base_url": _expand_env_placeholders(provider.get("base_url"), name, "base_url"),
        "model": _expand_env_placeholders(provider.get("model"), name, "model"),
        "endpoint": _expand_env_placeholders(provider.get("endpoint"), name, "endpoint"),
    }


def get_provider_mappings() -> dict:
    global _config
    if _config is None:
        _config = _load_config()
    return _config.get("provider_mappings", {})
