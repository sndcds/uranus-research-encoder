"""Explicit environment configuration. No dotenv discovery, ever."""

import os
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from .version import MODEL_REVISION


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    api_key: SecretStr
    model_root: Path = Path("/models")
    model: Literal["jina-v3"] = "jina-v3"
    model_revision: Literal[MODEL_REVISION] = MODEL_REVISION
    jina_noncommercial: bool = False
    max_concurrent_requests: int = Field(default=2, ge=1, le=8)
    enable_docs: bool = False

    @field_validator("api_key")
    @classmethod
    def valid_key(cls, value: SecretStr) -> SecretStr:
        key = value.get_secret_value()
        if not 32 <= len(key) <= 4096 or any(not 32 <= ord(c) <= 126 for c in key):
            raise ValueError("invalid_key")
        return value

    @field_validator("model_root")
    @classmethod
    def absolute_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("absolute_model_root_required")
        return value

    @model_validator(mode="after")
    def acknowledged_license(self) -> Self:
        if not self.jina_noncommercial:
            raise ValueError("license_acknowledgement_required")
        return self

    @classmethod
    def from_env(cls) -> Self:
        key_file = os.environ.get("ENCODER_API_KEY_FILE")
        if key_file:
            path = Path(key_file)
            if not path.is_absolute():
                raise ValueError("absolute_key_path_required")
            with path.open("rb") as stream:
                raw = stream.read(4098)
            # Permit one conventional final newline; never silently trim key spaces.
            key = raw.removesuffix(b"\n").removesuffix(b"\r").decode("ascii")
        else:
            key = os.environ.get("ENCODER_API_KEY", "")
        return cls(
            api_key=SecretStr(key),
            model_root=Path(os.environ.get("ENCODER_MODEL_ROOT", "/models")),
            model=os.environ.get("ENCODER_MODEL", "jina-v3"),
            model_revision=os.environ.get("ENCODER_MODEL_REVISION", MODEL_REVISION),
            jina_noncommercial=os.environ.get("JINA_NONCOMMERCIAL") == "1",
            max_concurrent_requests=os.environ.get("ENCODER_MAX_CONCURRENT_REQUESTS", "2"),
            enable_docs=os.environ.get("ENCODER_ENABLE_DOCS") == "1",
        )
