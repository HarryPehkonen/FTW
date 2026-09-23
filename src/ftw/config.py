"""ftw.toml loading and provider/tier resolution (ftw_plan.md §6).

API keys are read from the environment only, at the moment a provider is
built — never written into this config, into traces, or into events.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ftw.providers import IModelProvider, OpenAICompatibleProvider


class ConfigError(Exception):
    pass


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["openai_compatible"]
    base_url: str
    api_key_env: str | None = None


class TierConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str


class FtwConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    providers: dict[str, ProviderConfig] = {}
    tiers: dict[str, TierConfig] = {}


def load_config(path: str | Path) -> FtwConfig:
    path = Path(path)
    if not path.exists():
        return FtwConfig()
    with path.open("rb") as f:
        data = tomllib.load(f)
    return FtwConfig.model_validate(data)


def build_provider(config: FtwConfig, tier: str) -> IModelProvider:
    tier_cfg = config.tiers.get(tier)
    if tier_cfg is None:
        raise ConfigError(f"unknown model tier {tier!r} (configured tiers: {sorted(config.tiers)})")

    provider_cfg = config.providers.get(tier_cfg.provider)
    if provider_cfg is None:
        raise ConfigError(
            f"tier {tier!r} references unknown provider {tier_cfg.provider!r} "
            f"(configured providers: {sorted(config.providers)})"
        )

    api_key = None
    if provider_cfg.api_key_env is not None:
        api_key = os.environ.get(provider_cfg.api_key_env)
        if not api_key:
            raise ConfigError(
                f"provider {tier_cfg.provider!r} needs ${provider_cfg.api_key_env} set in the environment"
            )

    if provider_cfg.kind == "openai_compatible":
        return OpenAICompatibleProvider(base_url=provider_cfg.base_url, api_key=api_key, model=tier_cfg.model)

    raise ConfigError(f"unsupported provider kind {provider_cfg.kind!r}")
