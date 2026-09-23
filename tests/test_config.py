"""ftw.toml loading and provider/tier resolution (ftw_plan.md §6).

API keys come from the environment only — never written to config, traces,
or events, and never silently sent as an empty/missing auth header without
a clear error.
"""

import pytest

from ftw.config import ConfigError, build_provider, load_config
from ftw.providers import OpenAICompatibleProvider


class TestLoadConfig:
    def test_missing_file_returns_empty_config(self, tmp_path):
        config = load_config(tmp_path / "does-not-exist.toml")
        assert config.providers == {}
        assert config.tiers == {}

    def test_loads_providers_and_tiers(self, tmp_path):
        path = tmp_path / "ftw.toml"
        path.write_text(
            """
            [providers.deepseek]
            kind = "openai_compatible"
            base_url = "https://api.deepseek.com"
            api_key_env = "DEEPSEEK_API_KEY"

            [providers.local]
            kind = "openai_compatible"
            base_url = "http://localhost:8080/v1"

            [tiers.fast]
            provider = "deepseek"
            model = "deepseek-chat"

            [tiers.local]
            provider = "local"
            model = "whatever-is-loaded"
            """
        )

        config = load_config(path)

        assert config.providers["deepseek"].base_url == "https://api.deepseek.com"
        assert config.providers["deepseek"].api_key_env == "DEEPSEEK_API_KEY"
        assert config.providers["local"].api_key_env is None
        assert config.tiers["fast"].model == "deepseek-chat"


class TestBuildProvider:
    def test_builds_openai_compatible_provider_with_key_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")
        path = tmp_path / "ftw.toml"
        path.write_text(
            """
            [providers.deepseek]
            kind = "openai_compatible"
            base_url = "https://api.deepseek.com"
            api_key_env = "DEEPSEEK_API_KEY"

            [tiers.fast]
            provider = "deepseek"
            model = "deepseek-chat"
            """
        )
        config = load_config(path)

        provider = build_provider(config, "fast")

        assert isinstance(provider, OpenAICompatibleProvider)

    def test_provider_with_no_api_key_env_needs_no_key(self, tmp_path):
        path = tmp_path / "ftw.toml"
        path.write_text(
            """
            [providers.local]
            kind = "openai_compatible"
            base_url = "http://localhost:8080/v1"

            [tiers.local]
            provider = "local"
            model = "m"
            """
        )
        config = load_config(path)
        provider = build_provider(config, "local")
        assert isinstance(provider, OpenAICompatibleProvider)

    def test_missing_env_var_raises_clear_error(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NOUS_API_KEY", raising=False)
        path = tmp_path / "ftw.toml"
        path.write_text(
            """
            [providers.nous]
            kind = "openai_compatible"
            base_url = "https://inference-api.nousresearch.com/v1"
            api_key_env = "NOUS_API_KEY"

            [tiers.smart]
            provider = "nous"
            model = "whatever"
            """
        )
        config = load_config(path)

        with pytest.raises(ConfigError, match="NOUS_API_KEY"):
            build_provider(config, "smart")

    def test_unknown_tier_raises(self, tmp_path):
        config = load_config(tmp_path / "missing.toml")
        with pytest.raises(ConfigError, match="fast"):
            build_provider(config, "fast")

    def test_tier_referencing_unknown_provider_raises(self, tmp_path):
        path = tmp_path / "ftw.toml"
        path.write_text(
            """
            [tiers.fast]
            provider = "ghost"
            model = "m"
            """
        )
        config = load_config(path)
        with pytest.raises(ConfigError, match="ghost"):
            build_provider(config, "fast")
