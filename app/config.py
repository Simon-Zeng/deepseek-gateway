"""Gateway configuration via environment variables and YAML files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve project root (one level up from app/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_yaml(path: Path) -> dict:
    """Load a YAML file if it exists, return empty dict otherwise."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


class DeepSeekSettings(BaseSettings):
    """DeepSeek API connection settings."""

    model_config = SettingsConfigDict(env_prefix="DEEPSEEK_")

    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    timeout: int = 300
    max_retries: int = 2
    retry_delay: float = 1.0
    connection_pool_size: int = 100


class GatewaySettings(BaseSettings):
    """Gateway server settings."""

    model_config = SettingsConfigDict(env_prefix="GATEWAY_")

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    log_level: str = "info"

    # Auth
    api_key: str = ""  # Gateway-level API key (optional)
    api_key_forwarding: bool = True  # Forward client key to DeepSeek if no gateway key

    # Streaming
    stream_timeout: int = 300  # Max streaming duration in seconds
    ping_interval: int = 15  # Anthropic SSE ping interval

    # Rate limiting
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 100_000

    # Reasoning handling for OpenAI Chat
    reasoning_mode: str = "drop"  # drop | prepend | custom_field
    reasoning_prepend_marker: str = "<think>\n</think>"

    # Model mapping config path
    model_mapping_path: str = str(PROJECT_ROOT / "config" / "model_mapping.yaml")
    gateway_config_path: str = str(PROJECT_ROOT / "config" / "gateway.yaml")


class Settings(BaseSettings):
    """Root settings aggregating all sub-settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    deepseek: DeepSeekSettings = Field(default_factory=DeepSeekSettings)
    gateway: GatewaySettings = Field(default_factory=GatewaySettings)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override from gateway.yaml if it exists
        self._apply_yaml_overrides()

    def _apply_yaml_overrides(self):
        """Apply overrides from gateway.yaml config file."""
        gateway_yaml = _load_yaml(Path(self.gateway.gateway_config_path))
        if not gateway_yaml:
            return

        # DeepSeek overrides
        ds = gateway_yaml.get("deepseek", {})
        if ds.get("base_url"):
            self.deepseek.base_url = ds["base_url"]
        if ds.get("timeout"):
            self.deepseek.timeout = ds["timeout"]
        if ds.get("max_retries"):
            self.deepseek.max_retries = ds["max_retries"]
        if ds.get("retry_delay"):
            self.deepseek.retry_delay = ds["retry_delay"]
        if ds.get("connection_pool_size"):
            self.deepseek.connection_pool_size = ds["connection_pool_size"]

        # Server overrides
        srv = gateway_yaml.get("server", {})
        if srv.get("host"):
            self.gateway.host = srv["host"]
        if srv.get("port"):
            self.gateway.port = srv["port"]
        if srv.get("workers"):
            self.gateway.workers = srv["workers"]

        # Logging overrides
        log = gateway_yaml.get("logging", {})
        if log.get("level"):
            self.gateway.log_level = log["level"]


# Singleton settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance (lazy-initialized)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Force-reload settings (useful for testing)."""
    global _settings
    _settings = Settings()
    return _settings
