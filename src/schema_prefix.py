import json
import re

from pydantic import BaseModel, ConfigDict

from src.json_prefix import (
    IncompleteJSON,
    InvalidJSONPrefix,
    JSONPrefixParser,
    is_valid_json_prefix,
)
from src.models import FunctionDefinition


class SchemaPrefixValidator(BaseModel):
    """Validate partial function calls against available definitions.

    Args:
        functions: Allowed function signatures used to validate prefixes.
    """

    model_config = ConfigDict(extra="forbid")

    functions: dict[str, FunctionDefinition]

    def __init__(
        self,
        functions: list[FunctionDefinition],
    ) -> None:
        """Build lookup structures for prefix and schema validation.

        Args:
            functions: Available function definitions.
        """

        function_map = {
            function.name: function
            for function in functions
        }

        if len(function_map) != len(functions):
            raise ValueError("Duplicate function names are not allowed.")

        super().__init__(functions=function_map)

    def is_valid_function_name_prefix(
        self,
        prefix: str,
    ) -> bool:
        """Check whether a function-name prefix is still valid.

        Args:
            prefix: Partial function name.

        Returns:
            ``True`` if any known function starts with ``prefix``.
        """

        return any(
            name.startswith(prefix)
            for name in self.functions
        )

    def get_function(
        self,
        name: str,
    ) -> FunctionDefinition | None:
        """Get a function definition by name.

        Args:
            name: Exact function name.

        Returns:
            Matching function definition, or ``None`` if unknown.
        """

        return self.functions.get(name)

    def get_parameter_type(
        self,
        function_name: str,
        parameter_name: str,
    ) -> str | None:
        """Get the expected type of a function parameter.

        Args:
            function_name: Function owning the parameter.
            parameter_name: Parameter name to resolve.

        Returns:
            JSON type string for the parameter, or ``None`` if unknown.
        """

        function = self.get_function(function_name)

        if function is None:
            return None

        parameter = function.parameters.get(parameter_name)

        if parameter is None:
            return None

        return parameter.type

    def is_valid_parameter_prefix(
        self,
        function_name: str,
        prefix: str,
        used_parameters: set[str] | None = None,
    ) -> bool:
        """Check whether a parameter-name prefix is still valid.

        Args:
            function_name: Selected function name.
            prefix: Partial parameter name.
            used_parameters: Parameters already emitted in JSON.

        Returns:
            ``True`` if the prefix can still match an unused parameter.
        """

        function = self.get_function(function_name)

        if function is None:
            return False

        used = used_parameters or set()

        return any(
            name.startswith(prefix) and name not in used
            for name in function.parameters
        )

    def is_valid_value_prefix(
        self,
        expected_type: str,
        prefix: str,
    ) -> bool:
        """Check whether text can become a value of the expected type.

        Args:
            expected_type: Expected JSON type.
            prefix: Partial JSON value text.

        Returns:
            ``True`` if the prefix can still become a valid typed value.
        """

        text = prefix.lstrip()

        if not text:
            return True

        if not is_valid_json_prefix(text):
            return False

        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return self._is_possible_incomplete_value(
                expected_type,
                text,
            )

        return self._matches_type(value, expected_type)

    @staticmethod
    def _is_possible_incomplete_value(
        expected_type: str,
        text: str,
    ) -> bool:
        """Check whether a truncated value can still match a target type.

        Args:
            expected_type: Expected JSON type.
            text: Truncated JSON value text.

        Returns:
            ``True`` if continuing ``text`` may produce a valid value.
        """

        if not text:
            return True

        if expected_type == "string":
            return text.startswith('"')

        if expected_type == "number":
            return text[0] == "-" or "0" <= text[0] <= "9"

        if expected_type == "integer":
            return bool(
                re.fullmatch(
                    r"-?(?:0|[1-9][0-9]*)?",
                    text,
                )
            )

        if expected_type == "boolean":
            return text[0] in "tf"

        if expected_type == "null":
            return text[0] == "n"

        if expected_type == "object":
            return text[0] == "{"

        if expected_type == "array":
            return text[0] == "["

        return False

    @staticmethod
    def _matches_type(
        value: object,
        expected_type: str,
    ) -> bool:
        """Check runtime value compatibility with a JSON type.

        Args:
            value: Decoded runtime value.
            expected_type: Expected JSON type.

        Returns:
            ``True`` when the value matches the expected type.
        """

        if expected_type == "string":
            return isinstance(value, str)

        if expected_type == "number":
            return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            )

        if expected_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)

        if expected_type == "boolean":
            return isinstance(value, bool)

        if expected_type == "object":
            return isinstance(value, dict)

        if expected_type == "array":
            return isinstance(value, list)

        if expected_type == "null":
            return value is None

        return False

    def validate_call(self, data: object) -> bool:
        """Validate a complete function call against schema rules.

        Args:
            data: Candidate function-call object.

        Returns:
            ``True`` when object matches name, keys, and parameter types.
        """

        if not isinstance(data, dict):
            return False

        if set(data) != {"name", "parameters"}:
            return False

        name = data["name"]
        parameters = data["parameters"]

        if not isinstance(name, str):
            return False

        function = self.get_function(name)

        if function is None:
            return False

        if not isinstance(parameters, dict):
            return False

        expected = frozenset(function.parameters)
        received = set(parameters)

        if expected != frozenset(received):
            return False

        for parameter_name, definition in function.parameters.items():
            value = parameters[parameter_name]

            if not self._matches_type(value, definition.type):
                return False

        return True

    @staticmethod
    def _read_expected_string(
        parser: JSONPrefixParser,
        expected: str,
    ) -> None:
        """Read a property string and enforce exact expected content.

        Args:
            parser: Prefix parser positioned at a JSON string.
            expected: Required string value.

        Raises:
            IncompleteJSON: If more text is needed.
            InvalidJSONPrefix: If parsed value cannot match ``expected``.
        """

        if parser._peek() != '"':
            raise InvalidJSONPrefix()

        start = parser.position

        try:
            parser._parse_string()
        except IncompleteJSON:
            partial = parser.text[start + 1:]

            if "\\" in partial or not expected.startswith(partial):
                raise InvalidJSONPrefix() from None

            raise

        try:
            value = json.loads(parser.text[start:parser.position])
        except json.JSONDecodeError:
            raise InvalidJSONPrefix() from None

        if value != expected:
            raise InvalidJSONPrefix()

    def is_valid_call_prefix(self, text: str) -> bool:
        """Check whether text can still become a valid function call JSON.

        Args:
            text: Partial or complete JSON text.

        Returns:
            ``True`` when text is a valid prefix of a schema-valid call.
        """

        parser = JSONPrefixParser(text=text)

        try:
            parser._skip_whitespace()

            if parser._peek() is None:
                return True

            if parser._peek() != "{":
                raise InvalidJSONPrefix()

            parser._consume()
            parser._skip_whitespace()

            if parser._peek() is None:
                return True

            self._read_expected_string(parser, "name")

            parser._skip_whitespace()

            if parser._peek() is None:
                return True

            if parser._peek() != ":":
                raise InvalidJSONPrefix()

            parser._consume()
            parser._skip_whitespace()

            if parser._peek() is None:
                return True

            function_name = self._read_function_name(parser)

            if function_name is None:
                return False

            parser._skip_whitespace()

            if parser._peek() is None:
                return True

            if parser._peek() != ",":
                raise InvalidJSONPrefix()

            parser._consume()
            parser._skip_whitespace()

            if parser._peek() is None:
                return True

            self._read_expected_string(parser, "parameters")

            parser._skip_whitespace()

            if parser._peek() is None:
                return True

            if parser._peek() != ":":
                raise InvalidJSONPrefix()

            parser._consume()
            parser._skip_whitespace()

            if parser._peek() is None:
                return True

            if parser._peek() != "{":
                raise InvalidJSONPrefix()

            parser._consume()

            return self._parse_parameters(
                parser,
                function_name,
            )

        except IncompleteJSON:
            return True

        except InvalidJSONPrefix:
            return False

    @staticmethod
    def _read_complete_string(
        parser: JSONPrefixParser,
    ) -> str | None:
        """Read and decode a complete JSON string token.

        Args:
            parser: Prefix parser at string start.

        Returns:
            Decoded string value, or ``None`` if decoding is not a string.

        Raises:
            InvalidJSONPrefix: If the string cannot be decoded.
        """

        start = parser.position

        parser._parse_string()

        try:
            value = json.loads(parser.text[start:parser.position])
        except json.JSONDecodeError:
            raise InvalidJSONPrefix() from None

        if not isinstance(value, str):
            return None

        return value

    def _read_function_name(
        self,
        parser: JSONPrefixParser,
    ) -> str | None:
        """Read function name and validate it against known definitions.

        Args:
            parser: Prefix parser positioned at function-name string.

        Returns:
            Valid function name when complete.

        Raises:
            IncompleteJSON: If name token is still incomplete.
            InvalidJSONPrefix: If name cannot become a known function.
        """

        if parser._peek() != '"':
            raise InvalidJSONPrefix()

        start = parser.position
        parser._consume()

        while True:
            char = parser._peek()

            if char is None:
                partial = parser.text[start + 1:]

                if not self.is_valid_function_name_prefix(partial):
                    raise InvalidJSONPrefix()

                raise IncompleteJSON()

            if char == '"':
                parser.position += 1
                break

            if char == "\\" or ord(char) < 0x20:
                raise InvalidJSONPrefix()

            parser.position += 1

        try:
            name = json.loads(parser.text[start:parser.position])
        except json.JSONDecodeError:
            raise InvalidJSONPrefix() from None

        if not isinstance(name, str):
            raise InvalidJSONPrefix()

        if name not in self.functions:
            raise InvalidJSONPrefix()

        return name

    def _parse_parameters(
        self,
        parser: JSONPrefixParser,
        function_name: str,
    ) -> bool:
        """Parse the ``parameters`` object of a function call prefix.

        Args:
            parser: Prefix parser inside ``parameters`` object.
            function_name: Selected function name.

        Returns:
            ``True`` when the parameters prefix remains valid.

        Raises:
            InvalidJSONPrefix: If structure is irrecoverably invalid.
        """

        function = self.get_function(function_name)

        if function is None:
            return False

        used: set[str] = set()
        expected_parameters = set(function.parameters)

        parser._skip_whitespace()

        if parser._peek() is None:
            return True

        if parser._peek() == "}":
            if function.parameters:
                return False

            parser.position += 1
            return self._parse_call_end(parser)

        while True:
            parser._skip_whitespace()

            if parser._peek() is None:
                return True

            if parser._peek() != '"':
                raise InvalidJSONPrefix()

            key_start = parser.position

            try:
                parameter_name = self._read_complete_string(parser)
            except IncompleteJSON:
                partial = parser.text[key_start + 1:]

                if self.is_valid_parameter_prefix(
                    function_name,
                    partial,
                    used,
                ):
                    return True

                return False

            if parameter_name is None:
                return False

            if parameter_name not in function.parameters:
                return False

            if parameter_name in used:
                return False

            used.add(parameter_name)

            parser._skip_whitespace()

            if parser._peek() is None:
                return True

            if parser._peek() != ":":
                raise InvalidJSONPrefix()

            parser._consume()
            parser._skip_whitespace()

            if parser._peek() is None:
                return True

            value_start = parser.position
            expected_type = function.parameters[parameter_name].type

            try:
                parser._parse_value()
            except IncompleteJSON:
                partial_value = parser.text[value_start:]

                return self.is_valid_value_prefix(
                    expected_type,
                    partial_value,
                )

            value_text = parser.text[value_start:parser.position]

            try:
                value = json.loads(value_text)
            except json.JSONDecodeError:
                return False

            if not self._matches_type(value, expected_type):
                return False

            parser._skip_whitespace()

            if parser._peek() is None:
                return True

            char = parser._peek()

            if char == "}":
                if used != expected_parameters:
                    return False

                parser.position += 1
                return self._parse_call_end(parser)

            if char != ",":
                raise InvalidJSONPrefix()

            parser._consume()
            parser._skip_whitespace()

            if parser._peek() is None:
                return True

    @staticmethod
    def _parse_call_end(
        parser: JSONPrefixParser,
    ) -> bool:
        """Validate the trailing part after ``parameters`` is closed.

        Args:
            parser: Prefix parser after ``parameters`` object.

        Returns:
            ``True`` when remaining text is valid or validly incomplete.
        """

        parser._skip_whitespace()

        if parser._peek() is None:
            return True

        if parser._peek() != "}":
            return False

        parser._consume()
        parser._skip_whitespace()

        return parser._peek() is None
