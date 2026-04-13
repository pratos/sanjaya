"""User-configurable prompt configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from typing import Any


@dataclass(frozen=True)
class PromptConfig:
    """Configuration for overridable prompts.

    Every field defaults to None, meaning "use the built-in default."
    Set a field to a string to override that prompt entirely.

    The answer_schema field accepts a dict matching the schema format
    returned by generate_answer_schema(). When set, the schema generation
    LLM call is skipped entirely.
    """

    # Toolkit strategy prompts
    video_strategy: str | None = None
    video_vision_first_strategy: str | None = None
    document_strategy: str | None = None
    image_strategy: str | None = None

    # Critic
    critic: str | None = None

    # Answer schema (dict, not str — skips LLM call when provided)
    answer_schema: dict[str, Any] | None = field(default=None, repr=False)

    def with_overrides(self, **kwargs: Any) -> PromptConfig:
        """Return a new PromptConfig with the given fields replaced."""
        current = {f.name: getattr(self, f.name) for f in fields(self)}
        current.update(kwargs)
        return PromptConfig(**current)

    def to_dict(self) -> dict[str, str]:
        """Serialize to dict[str, str] for GEPA compatibility.

        Only includes non-None fields. answer_schema is JSON-encoded.
        """
        result: dict[str, str] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val is None:
                continue
            if isinstance(val, dict):
                result[f.name] = json.dumps(val)
            else:
                result[f.name] = val
        return result

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> PromptConfig:
        """Deserialize from dict[str, str] (GEPA candidate format)."""
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name in d:
                val = d[f.name]
                if f.name == "answer_schema":
                    kwargs[f.name] = json.loads(val)
                else:
                    kwargs[f.name] = val
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str) -> PromptConfig:
        """Load from a YAML file. Keys match field names."""
        import yaml

        with open(path) as fh:
            data = yaml.safe_load(fh)
        return cls(**{k: v for k, v in data.items() if k in {f.name for f in fields(cls)}})
