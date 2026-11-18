
from pydantic import BaseModel, ConfigDict, Field, model_validator


InvalidJSONPrefix = SyntaxError
IncompleteJSON = EOFError


class JSONPrefixParser(BaseModel):
    """Parse JSON incrementally to validate incomplete JSON prefixes.

    Attributes:
        text: JSON text being parsed.
        position: Current parsing position in the input text.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    position: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_position(self) -> "JSONPrefixParser":
        """Validate that the parser position is within the input text.

        Returns:
            The validated parser instance.

        Raises:
            ValueError: If the position exceeds the input length.
        """
        if self.position > len(self.text):
            raise ValueError("Parser position exceeds input length.")

        return self

    def _peek(self) -> str | None:
        """Return the current character without advancing the position.

        Returns:
            The current character, or None if the input has ended.
        """
        if self.position >= len(self.text):
            return None

        return self.text[self.position]

    def _consume(self) -> str:
        """Consume and return the current character.

        Returns:
            The consumed character.

        Raises:
            EOFError: If the input has ended.
        """
        char = self._peek()

        if char is None:
            raise IncompleteJSON("Unexpected end of input.")

        self.position += 1
        return char

    def _skip_whitespace(self) -> None:
        """Skip whitespace characters allowed by the JSON specification."""
        while self._peek() in (" ", "\t", "\n", "\r"):
            self.position += 1

    def _parse_string(self) -> None:
        """Parse a JSON string.

        Raises:
            SyntaxError: If the string contains invalid syntax.
            EOFError: If the string is incomplete.
        """
        if self._consume() != '"':
            raise InvalidJSONPrefix("Expected string.")

        while True:
            char = self._consume()

            if char == '"':
                return

            if char == "\\":
                escaped = self._consume()

                if escaped not in '"\\/bfnrtu':
                    raise InvalidJSONPrefix("Invalid escape sequence.")

                if escaped == "u":
                    for _ in range(4):
                        hex_char = self._consume()

                        if hex_char not in "0123456789abcdefABCDEF":
                            raise InvalidJSONPrefix(
                                "Invalid Unicode escape."
                            )

            elif ord(char) < 0x20:
                raise InvalidJSONPrefix("Unescaped control character.")

    def _parse_number(self) -> None:
        """Parse a JSON number.

        Raises:
            SyntaxError: If the number contains invalid syntax.
            EOFError: If the number is incomplete.
        """
        if self._peek() == "-":
            self.position += 1

        char = self._peek()

        if char is None:
            raise IncompleteJSON("Incomplete number.")

        if char == "0":
            self.position += 1
            next_char = self._peek()

            if next_char is not None and "0" <= next_char <= "9":
                raise InvalidJSONPrefix("Leading zeros are not allowed.")

        elif "1" <= char <= "9":
            while True:
                next_char = self._peek()

                if next_char is None or not "0" <= next_char <= "9":
                    break

                self.position += 1

        else:
            raise InvalidJSONPrefix("Invalid number.")

        if self._peek() == ".":
            self.position += 1

            char = self._peek()

            if char is None:
                raise IncompleteJSON("Incomplete decimal number.")

            if not "0" <= char <= "9":
                raise InvalidJSONPrefix("Invalid decimal number.")

            while True:
                next_char = self._peek()

                if next_char is None or not "0" <= next_char <= "9":
                    break

                self.position += 1

        if self._peek() in ("e", "E"):
            self.position += 1

            if self._peek() in ("+", "-"):
                self.position += 1

            char = self._peek()

            if char is None:
                raise IncompleteJSON("Incomplete exponent.")

            if not "0" <= char <= "9":
                raise InvalidJSONPrefix("Invalid exponent.")

            while True:
                next_char = self._peek()

                if next_char is None or not "0" <= next_char <= "9":
                    break

                self.position += 1

    def _parse_literal(self, literal: str) -> None:
        """Parse a JSON literal.

        Args:
            literal: Expected JSON literal.

        Raises:
            SyntaxError: If the literal does not match the expected value.
            EOFError: If the literal is incomplete.
        """
        for expected in literal:
            char = self._consume()

            if char != expected:
                raise InvalidJSONPrefix("Invalid literal.")

    def _parse_array(self) -> None:
        """Parse a JSON array.

        Raises:
            SyntaxError: If the array contains invalid syntax.
            EOFError: If the array is incomplete.
        """
        self._consume()
        self._skip_whitespace()

        if self._peek() == "]":
            self.position += 1
            return

        while True:
            self._parse_value()
            self._skip_whitespace()

            char = self._peek()

            if char == "]":
                self.position += 1
                return

            if char != ",":
                if char is None:
                    raise IncompleteJSON("Incomplete array.")

                raise InvalidJSONPrefix("Expected ',' or ']'.")

            self.position += 1
            self._skip_whitespace()

    def _parse_object(self) -> None:
        """Parse a JSON object.

        Raises:
            SyntaxError: If the object contains invalid syntax.
            EOFError: If the object is incomplete.
        """
        self._consume()
        self._skip_whitespace()

        if self._peek() == "}":
            self.position += 1
            return

        while True:
            if self._peek() != '"':
                if self._peek() is None:
                    raise IncompleteJSON("Incomplete object.")

                raise InvalidJSONPrefix("Expected object key.")

            self._parse_string()
            self._skip_whitespace()

            if self._peek() != ":":
                if self._peek() is None:
                    raise IncompleteJSON("Expected ':'.")

                raise InvalidJSONPrefix("Expected ':'.")

            self.position += 1
            self._skip_whitespace()
            self._parse_value()
            self._skip_whitespace()

            char = self._peek()

            if char == "}":
                self.position += 1
                return

            if char != ",":
                if char is None:
                    raise IncompleteJSON("Incomplete object.")

                raise InvalidJSONPrefix("Expected ',' or '}'.")

            self.position += 1
            self._skip_whitespace()

    def _parse_value(self) -> None:
        """Parse a JSON value.

        Raises:
            SyntaxError: If the value contains invalid syntax.
            EOFError: If the value is incomplete.
        """
        self._skip_whitespace()
        char = self._peek()

        if char is None:
            raise IncompleteJSON("Expected JSON value.")

        if char == '"':
            self._parse_string()

        elif char == "{":
            self._parse_object()

        elif char == "[":
            self._parse_array()

        elif char == "t":
            self._parse_literal("true")

        elif char == "f":
            self._parse_literal("false")

        elif char == "n":
            self._parse_literal("null")

        elif char == "-" or "0" <= char <= "9":
            self._parse_number()

        else:
            raise InvalidJSONPrefix("Invalid JSON value.")

    def is_valid_prefix(self) -> bool:
        """Check whether the input is a valid or incomplete JSON prefix.

        Returns:
            True if the input is a valid JSON prefix, otherwise False.
        """
        try:
            self._parse_value()
            self._skip_whitespace()

            return self.position == len(self.text)

        except IncompleteJSON:
            return True

        except InvalidJSONPrefix:
            return False


def is_valid_json_prefix(text: str) -> bool:
    """Check whether the provided text is a valid JSON prefix.

    Args:
        text: JSON text to validate.

    Returns:
        True if the text is a valid or incomplete JSON prefix,
        otherwise False.
    """
    parser = JSONPrefixParser(text=text)
    return parser.is_valid_prefix()
