"""Provider-independent representation of a recorded agent tool call."""

import json
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import TypedDict


class ToolCallMismatchKind(StrEnum):
    """Why a recorded trace differs from the expected trace."""

    WRONG_TOOL_CALL = "wrong_tool_call"
    WRONG_TOOL_ARGUMENTS = "wrong_tool_arguments"
    MISSING_TOOL_CALL = "missing_tool_call"
    EXTRA_TOOL_CALL = "extra_tool_call"
    MISSING_EXPECTED_TRACE = "missing_expected_trace"
    MISSING_ACTUAL_TRACE = "missing_actual_trace"


class ToolCallMismatch(TypedDict):
    """A discrepancy and its one-based step, if a step can be identified."""

    kind: ToolCallMismatchKind
    step: Annotated[int, Field(gt=0, strict=True)] | None


class ToolCall(BaseModel):
    """A tool name and its JSON arguments at one position in a trace."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, pattern=r"\S")
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("arguments")
    @classmethod
    def validate_arguments(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Use the same JSON representation for SDK and file-based traces."""
        try:
            normalized: dict[str, Any] = json.loads(json.dumps(value, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("Tool arguments must contain JSON values") from exc
        return normalized
