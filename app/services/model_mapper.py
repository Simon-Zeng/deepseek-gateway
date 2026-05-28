"""Model name mapping service."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

from app.models.common import ModelMappingConfig, ModelMappingRule, ModelType

logger = logging.getLogger(__name__)

# Reasoning effort levels in ascending order of strength
EFFORT_LEVELS = {"low": 0, "medium": 1, "high": 2, "xhigh": 3}

# Mapping from OpenAI/Anthropic reasoning_effort to DeepSeek format.
# DeepSeek only accepts: "low", "high", "none" (no "medium").
REASONING_EFFORT_TO_DEEPSEEK = {
    "low": "low",       # Direct mapping
    "medium": "high",   # DeepSeek has no "medium" — map up one level
    "high": "high",     # Direct mapping
    "max": "high",      # Anthropic Opus-only — map to highest DeepSeek value
}


def map_reasoning_effort(original_effort: str | None) -> str | None:
    """Map reasoning effort from OpenAI/Anthropic format to DeepSeek format.

    Provider effort definitions:
      OpenAI (o-series, GPT-5): low, medium, high
      Anthropic (adaptive thinking): low, medium, high, max (Opus-only)
      DeepSeek V4: low, high, none

    The key gap is "medium" — DeepSeek doesn't support it, so "medium" maps to "high".
    "max" (Anthropic Opus-only) also maps to "high" as DeepSeek's highest level.

    Args:
        original_effort: The effort value from the incoming request.

    Returns:
        DeepSeek-compatible effort value, or None if no effort was specified.
    """
    if not original_effort:
        return None
    normalized = original_effort.strip().lower()
    return REASONING_EFFORT_TO_DEEPSEEK.get(normalized)


class MappingResult(BaseModel):
    """Result of a model name mapping."""

    target_model: str  # DeepSeek model name to use
    model_type: ModelType  # chat or reasoner


class ModelMapper:
    """Maps incoming model names to DeepSeek model names based on configurable rules.

    Supports:
    - Regex-based model name mapping from YAML config
    - Reasoning effort override: when effort >= threshold, force use pro model
    """

    def __init__(self, config_path: str | Path):
        self._config_path = Path(config_path)
        self._config: ModelMappingConfig = ModelMappingConfig()
        self._compiled_patterns: list[tuple[re.Pattern, ModelMappingRule]] = []
        # Reasoning effort override settings
        self._effort_override_enabled: bool = True
        self._effort_threshold: str = "high"
        self._pro_model: str = "deepseek-v4-pro"
        self._load_config()

    def _load_config(self):
        """Load and compile mapping rules from YAML config."""
        if not self._config_path.exists():
            logger.warning("Model mapping config not found at %s, using defaults", self._config_path)
            self._setup_defaults()
            return

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            self._config = ModelMappingConfig(**raw)
            self._compile_patterns()

            # Load reasoning effort override settings
            effort_cfg = raw.get("reasoning_effort_override", {})
            self._effort_override_enabled = effort_cfg.get("enabled", True)
            self._effort_threshold = effort_cfg.get("threshold", "high")

            # Determine pro model from defaults
            defaults = raw.get("defaults", {})
            self._pro_model = defaults.get("reasoner", "deepseek-v4-pro")

            logger.info(
                "Loaded %d model mapping rules from %s (effort override: %s, threshold: %s)",
                len(self._config.mapping),
                self._config_path,
                "enabled" if self._effort_override_enabled else "disabled",
                self._effort_threshold,
            )
        except Exception as e:
            logger.error("Failed to load model mapping config: %s", e)
            self._setup_defaults()

    def _setup_defaults(self):
        """Set up default mapping rules when no config file exists."""
        self._config = ModelMappingConfig(
            mapping=[
                ModelMappingRule(pattern=r"^o[13](-mini|-preview)?$", target="deepseek-v4-pro", type=ModelType.REASONER),
                ModelMappingRule(pattern=r"^gpt-3\.5-turbo", target="deepseek-v4-pro", type=ModelType.REASONER),
                ModelMappingRule(pattern=r"^gpt-4-turbo$", target="deepseek-v4-pro", type=ModelType.REASONER),
                ModelMappingRule(pattern=r"^gpt-4", target="deepseek-v4-flash", type=ModelType.CHAT),
                ModelMappingRule(pattern=r"^claude.*opus", target="deepseek-v4-pro", type=ModelType.REASONER),
                ModelMappingRule(pattern=r"^claude", target="deepseek-v4-flash", type=ModelType.CHAT),
                ModelMappingRule(pattern=r"^deepseek-v4-", target="$0", type=ModelType.AUTO),
                ModelMappingRule(pattern=r"^deepseek-chat$", target="deepseek-v4-flash", type=ModelType.CHAT),
                ModelMappingRule(pattern=r"^deepseek-reasoner$", target="deepseek-v4-pro", type=ModelType.REASONER),
                ModelMappingRule(pattern=r".*", target="deepseek-v4-flash", type=ModelType.CHAT),
            ]
        )
        self._compile_patterns()
        self._pro_model = "deepseek-v4-pro"

    def _compile_patterns(self):
        """Pre-compile regex patterns for efficient matching."""
        self._compiled_patterns = []
        for rule in self._config.mapping:
            try:
                compiled = re.compile(rule.pattern)
                self._compiled_patterns.append((compiled, rule))
            except re.error as e:
                logger.warning("Invalid regex pattern '%s': %s", rule.pattern, e)

    def map_model(
        self,
        model_name: str,
        reasoning_effort: Optional[str] = None,
    ) -> MappingResult:
        """Map an incoming model name to a DeepSeek model name.

        Args:
            model_name: The model name from the incoming request.
            reasoning_effort: Optional reasoning effort level
                (provider-native values: "low", "medium", "high", "max").
                When effort >= threshold, force use of pro model regardless of
                mapping. The actual value-to-DeepSeek conversion (including
                ``medium→high`` mapping) is done by ``map_reasoning_effort()``
                in each converter's ``convert_request()`` — this method only
                uses the raw value for the override decision.

        Returns:
            MappingResult with the target DeepSeek model name and type.
        """
        # ── Reasoning effort override ──
        if self._effort_override_enabled and reasoning_effort:
            effort_lower = reasoning_effort.lower()
            threshold_lower = self._effort_threshold.lower()
            if effort_lower in EFFORT_LEVELS and threshold_lower in EFFORT_LEVELS:
                if EFFORT_LEVELS[effort_lower] >= EFFORT_LEVELS[threshold_lower]:
                    logger.info(
                        "Reasoning effort '%s' >= threshold '%s', forcing pro model: %s",
                        reasoning_effort, self._effort_threshold, self._pro_model,
                    )
                    return MappingResult(
                        target_model=self._pro_model,
                        model_type=ModelType.REASONER,
                    )

        # ── Normal mapping ──
        for compiled, rule in self._compiled_patterns:
            if compiled.match(model_name):
                # Support $0 replacement (use original model name)
                target = rule.target.replace("$0", model_name)

                # For AUTO type, determine based on target model name
                model_type = rule.type
                if model_type == ModelType.AUTO:
                    if "pro" in target or "reasoner" in target or "r1" in target:
                        model_type = ModelType.REASONER
                    else:
                        model_type = ModelType.CHAT

                logger.debug("Mapped model '%s' -> '%s' (type=%s)", model_name, target, model_type)
                return MappingResult(target_model=target, model_type=model_type)

        # Fallback (shouldn't reach here if catch-all pattern exists)
        logger.warning("No mapping rule matched model '%s', using deepseek-v4-flash", model_name)
        return MappingResult(target_model="deepseek-v4-flash", model_type=ModelType.CHAT)

    def is_thinking_model(self, model_name: str) -> bool:
        """Check if a DeepSeek model should emit thinking/reasoning blocks."""
        thinking_models = self._config.thinking.get("models", ["deepseek-v4-pro"])
        return model_name in thinking_models

    def get_thinking_min_length(self) -> int:
        """Get the minimum reasoning_content length to include thinking blocks."""
        return self._config.thinking.get("min_length", 0)

    @property
    def available_models(self) -> list[dict]:
        """Return a list of available models for the /v1/models endpoint."""
        models = []
        seen = set()
        for _, rule in self._compiled_patterns:
            if rule.target not in seen and not rule.target.startswith("$"):
                seen.add(rule.target)
                models.append({
                    "id": rule.target,
                    "object": "model",
                    "owned_by": "deepseek",
                })
        # Also add common aliases
        for alias in [
            "gpt-4o", "gpt-4", "gpt-3.5-turbo", "gpt-4-turbo",
            "o1", "o1-mini", "o3-mini",
            "claude-3-5-sonnet-20241022", "claude-3-opus-20240229",
            "claude-opus-4-latest", "claude-sonnet-4-latest",
        ]:
            if alias not in seen:
                result = self.map_model(alias)
                models.append({
                    "id": alias,
                    "object": "model",
                    "owned_by": "deepseek-gateway",
                })
        return models

    def reload(self):
        """Reload the mapping config from disk."""
        self._load_config()
