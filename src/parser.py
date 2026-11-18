
import json
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from src.models import FunctionDefinition, Prompt


def load_json(path: str | Path) -> object:
    """Load and parse JSON content from a file path.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If JSON content is invalid.
    """

    file_path = Path(path)

    try:
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Input file not found: {file_path}"
        ) from None

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {file_path}: {error}"
        ) from error


def load_prompts(path: str | Path) -> list[Prompt]:
    """Load and validate prompt objects from a JSON file.

    Args:
        path: Path to prompt input JSON.

    Returns:
        Validated list of prompt models.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If prompt schema validation fails.
    """

    data = load_json(path)

    try:
        return TypeAdapter(list[Prompt]).validate_python(data)

    except ValidationError as error:
        raise ValueError(
            f"Invalid prompt definitions: {error}"
        ) from error


def load_function_definitions(
    path: str | Path,
) -> list[FunctionDefinition]:
    """Load and validate function definitions from a JSON file.

    Args:
        path: Path to function-definition JSON.

    Returns:
        Validated list of function definitions.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If definitions are invalid or names are duplicated.
    """

    data = load_json(path)

    try:
        functions = TypeAdapter(
            list[FunctionDefinition]
        ).validate_python(data)

    except ValidationError as error:
        raise ValueError(
            f"Invalid function definitions: {error}"
        ) from error

    names = [function.name for function in functions]

    if len(names) != len(set(names)):
        raise ValueError("Duplicate function names found")

    return functions
