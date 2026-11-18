
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


JsonType = Literal[
    "string",
    "number",
    "integer",
    "boolean",
    "object",
    "array",
    "null",
]


class Prompt(BaseModel):
    """Prompt record loaded from input JSON.

    Attributes:
        prompt: Natural-language request to be converted into a function call.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)


class ParameterDefinition(BaseModel):
    """Parameter definition for a function signature."""

    model_config = ConfigDict(extra="forbid")

    type: JsonType
    description: str | None = None


class FunctionDefinition(BaseModel):
    """Function schema definition loaded from input JSON.

    Attributes:
        name: Function name that the model must output.
        description: Human-readable function description.
        parameters: Mapping of parameter names to type definitions.
        returns: Return-type definition metadata.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ParameterDefinition


class FunctionCall(BaseModel):
    """Normalized function call emitted by the generation pipeline.

    Attributes:
        prompt: Original user prompt.
        name: Selected function name.
        parameters: Generated argument object for the selected function.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, object]
