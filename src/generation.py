
import json
from typing import Any, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from src.models import FunctionDefinition
from src.schema_prefix import SchemaPrefixValidator


class TrieNode(BaseModel):
    """Node containing valid function-name continuations."""

    model_config = ConfigDict(extra="forbid")

    children: dict[int, "TrieNode"] = Field(default_factory=dict)
    function_name: str | None = None


class FunctionTrie(BaseModel):
    """Token trie for constrained function selection."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    model: Any
    prompt: str
    function_names: list[str] = Field(min_length=1)
    root: TrieNode = Field(default_factory=TrieNode)
    base_ids: list[int] = Field(default_factory=list)

    def __init__(
        self,
        model: Any,
        prompt: str,
        function_names: list[str],
    ) -> None:
        """Build a Trie from function names tokenized in their prompt context.

        Args:
            model: Model adapter exposing tokenization and logits APIs.
            prompt: Selection prompt preceding the function name.
            function_names: Available function names.

        Raises:
            ValueError: If tokenization changes the prompt prefix or
                function-name token sequences are ambiguous.
        """
        super().__init__(
            model=model,
            prompt=prompt,
            function_names=function_names,
        )

    def model_post_init(self, context: Any) -> None:
        """Build token sequences for all available function names.

        Args:
            context: Pydantic initialization context.

        Raises:
            ValueError: If tokenization changes the prompt prefix or
                function-name token sequences are ambiguous.
        """
        self.base_ids = self.model.encode(self.prompt)

        for name in self.function_names:
            candidate_ids = self.model.encode(f"{self.prompt}{name}\n")

            if candidate_ids[:len(self.base_ids)] != self.base_ids:
                raise ValueError(
                    "Function name tokenization changed the prompt prefix."
                )

            continuation = candidate_ids[len(self.base_ids):]

            if not continuation:
                raise ValueError(
                    f"Empty token sequence for function: {name}"
                )

            self._insert(continuation, name)

    def _insert(
        self,
        token_ids: list[int],
        function_name: str,
    ) -> None:
        """Insert a function-name token sequence into the Trie.

        Args:
            token_ids: Token sequence representing the function name.
            function_name: Function name associated with the sequence.

        Raises:
            ValueError: If the token sequence is ambiguous or duplicated.
        """
        node = self.root

        for token_id in token_ids:
            if node.function_name is not None:
                raise ValueError(
                    "Ambiguous function-name token sequences."
                )

            node = node.children.setdefault(token_id, TrieNode())

        if node.function_name is not None or node.children:
            raise ValueError(
                f"Ambiguous function-name token sequence: {function_name}"
            )

        node.function_name = function_name

    def select_function(
        self,
        model: Any,
        input_ids: list[int],
    ) -> str:
        """Select a function using constrained greedy decoding.

        Args:
            model: Model adapter exposing next-token logits.
            input_ids: Tokenized selection prompt.

        Returns:
            Name of the selected function.

        Raises:
            RuntimeError: If the model returns insufficient logits or
                selection does not reach a valid function.
            ValueError: If input ids do not match the Trie prompt.
        """
        if input_ids != self.base_ids:
            raise ValueError("Input ids do not match the Trie prompt.")

        node = self.root

        while node.function_name is None:
            if not node.children:
                raise RuntimeError(
                    "Function selection ended without a valid function."
                )

            logits = model.get_logits(input_ids)
            valid_tokens = list(node.children)

            if max(valid_tokens) >= len(logits):
                raise RuntimeError(
                    "Model returned insufficient logits for Trie tokens."
                )

            next_token_id = max(
                valid_tokens,
                key=lambda token_id: logits[token_id],
            )

            input_ids.append(next_token_id)
            node = node.children[next_token_id]

        return node.function_name


class ConstrainedGenerator(BaseModel):
    """Generate schema-compliant function calls token by token."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    model: Any
    functions: list[FunctionDefinition] = Field(min_length=1)
    max_new_tokens: int = Field(default=128, gt=0)

    _validator: SchemaPrefixValidator = PrivateAttr()
    _token_text_cache: dict[int, str | None] = PrivateAttr(
        default_factory=dict
    )
    _prefix_validity_cache: dict[str, bool] = PrivateAttr(
        default_factory=dict
    )
    _prompt_prefix: str = PrivateAttr(default="")

    def __init__(
        self,
        model: Any,
        functions: list[FunctionDefinition],
        max_new_tokens: int = 128,
    ) -> None:
        """Initialize constrained generation state.

        Args:
            model: Model adapter exposing encode/decode/logits APIs.
            functions: Available function definitions for schema enforcement.
            max_new_tokens: Maximum number of generated tokens per prompt.

        Raises:
            ValueError: If max_new_tokens is lower than 1 or no functions
                are provided.
        """
        super().__init__(
            model=model,
            functions=functions,
            max_new_tokens=max_new_tokens,
        )

    def model_post_init(self, context: Any) -> None:
        """Initialize private generation state.

        Args:
            context: Pydantic initialization context.
        """
        self._validator = SchemaPrefixValidator(self.functions)
        self._prompt_prefix = self._build_prompt_prefix()

    def _build_prompt_prefix(self) -> str:
        """Build the context containing all available function definitions.

        Returns:
            Text describing the available functions and their parameters.
        """
        definitions = []

        for function in self.functions:
            parameters = []

            for name, parameter in function.parameters.items():
                parameter_info = f"{name}: {parameter.type}"

                if parameter.description:
                    parameter_info += f" ({parameter.description})"

                parameters.append(parameter_info)

            parameter_text = ", ".join(parameters)

            definitions.append(
                f"{function.name}({parameter_text}): "
                f"{function.description}"
            )

        return (
            "Choose the function that best matches the user request. "
            "Interpret the complete request and identify the intended "
            "values for its parameters.\n"
            "When a parameter represents a regular expression:\n"
            "- Generate a pattern that matches the complete intended target.\n"
            "- Distinguish literal text from character classes and broader "
            "character categories.\n"
            "- Pay attention to quantifiers such as 'all' and 'any'.\n"
            "- Consider uppercase and lowercase characters when relevant.\n"
            "Do not simply copy a word or number from the request when a "
            "regular expression is needed.\n"
            "Available functions:\n"
            + "\n".join(definitions)
            + "\n"
        )

    def _build_selection_prompt(self, user_prompt: str) -> str:
        """Build the prompt used for constrained function selection.

        Args:
            user_prompt: Natural-language request from input data.

        Returns:
            Prompt containing function definitions and the user request.
        """
        return (
            self._prompt_prefix
            + f"User request: {user_prompt}\n"
            + "Selected function:\n"
        )

    def _build_prompt(
        self,
        user_prompt: str,
        function: FunctionDefinition,
    ) -> str:
        """Build the JSON generation prompt for the selected function.

        Args:
            user_prompt: Natural-language request from input data.
            function: Function selected by the Trie.

        Returns:
            Prompt containing the selected function and original request.
        """
        parameters = []

        for name, parameter in function.parameters.items():
            parameter_info = f"{name}: {parameter.type}"

            if parameter.description:
                parameter_info += f" ({parameter.description})"

            parameters.append(parameter_info)

        parameter_text = ", ".join(parameters)

        return (
            "Generate a function call using the selected function.\n"
            "Interpret the complete user request and generate the intended "
            "value for every parameter.\n"
            "When a parameter represents a regular expression:\n"
            "- Generate a pattern that matches the complete intended target.\n"
            "- Distinguish literal text from character classes and broader "
            "character categories.\n"
            "- Pay attention to quantifiers such as 'all' and 'any'.\n"
            "- Consider uppercase and lowercase characters when relevant.\n"
            "Do not simply copy a word or number from the request when a "
            "regular expression is needed.\n"
            'Return only a JSON object with keys "name" and "parameters".\n'
            f"Function: {function.name}\n"
            f"Description: {function.description}\n"
            f"Parameters: {parameter_text}\n"
            f"User request: {user_prompt}\n"
            "JSON:"
        )

    def _is_valid_prefix(
        self,
        text: str,
        validator: SchemaPrefixValidator,
    ) -> bool:
        """Validate and cache JSON prefixes.

        Args:
            text: Candidate partial JSON text.
            validator: Schema validator for the selected function.

        Returns:
            True when the prefix can still become a valid function call.
        """
        cached = self._prefix_validity_cache.get(text)

        if cached is not None:
            return cached

        is_valid = validator.is_valid_call_prefix(text)

        if len(self._prefix_validity_cache) > 100_000:
            self._prefix_validity_cache.clear()

        self._prefix_validity_cache[text] = is_valid

        return is_valid

    def _get_token_text(self, token_id: int) -> str | None:
        """Decode and cache a token string.

        Args:
            token_id: Token id to decode.

        Returns:
            Decoded token text, or None when decoding fails.
        """
        if token_id not in self._token_text_cache:
            try:
                self._token_text_cache[token_id] = self.model.decode(
                    [token_id]
                )
            except (IndexError, KeyError, ValueError):
                self._token_text_cache[token_id] = None

        return self._token_text_cache[token_id]

    def _get_next_token(
        self,
        input_ids: list[int],
        current_text: str,
        validator: SchemaPrefixValidator,
    ) -> tuple[int, str, str]:
        """Select the best valid next token.

        Args:
            input_ids: Current input token ids passed to the model.
            current_text: Generated JSON text so far.
            validator: Schema validator for the selected function.

        Returns:
            A tuple containing the accepted token id, updated text,
            and decoded token text.

        Raises:
            RuntimeError: If no candidate token keeps a valid prefix.
        """
        logits = np.asarray(
            self.model.get_logits(input_ids),
            dtype=np.float64,
        )

        vocab_size = int(logits.shape[0])

        if vocab_size == 0:
            raise RuntimeError("Model returned empty logits.")

        seen: set[int] = set()
        current_k = min(128, vocab_size)

        while current_k <= vocab_size:
            top_candidates = np.argpartition(
                logits,
                -current_k,
            )[-current_k:]

            top_candidates = top_candidates[
                np.argsort(logits[top_candidates])[::-1]
            ]

            for candidate in top_candidates:
                token_id = int(candidate)

                if token_id in seen:
                    continue

                seen.add(token_id)

                token_text = self._get_token_text(token_id)

                if not token_text:
                    continue

                candidate_text = current_text + token_text

                if not self._is_valid_prefix(candidate_text, validator):
                    continue

                return token_id, candidate_text, token_text

            if current_k == vocab_size:
                break

            current_k = min(current_k * 2, vocab_size)

        raise RuntimeError(
            "No valid next token found for the current JSON prefix."
        )

    @staticmethod
    def _normalize_parameters(
        parameters: dict[str, object],
        function: FunctionDefinition,
    ) -> dict[str, object]:
        """Normalize number parameters to float values.

        Args:
            parameters: Generated function parameters.
            function: Function definition containing parameter types.

        Returns:
            Parameters with number values normalized to floats.
        """
        normalized: dict[str, object] = {}

        for name, value in parameters.items():
            parameter = function.parameters[name]

            if (
                parameter.type == "number"
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                normalized[name] = float(value)
            else:
                normalized[name] = value

        return normalized

    def generate(self, user_prompt: str) -> dict[str, object]:
        """Generate a single schema-valid function call.

        Args:
            user_prompt: Natural-language request to transform into a call.

        Returns:
            A dictionary containing the selected function name and parameters.

        Raises:
            ValueError: If the input prompt is empty.
            RuntimeError: If function selection or generation fails, or
                if the generated call violates schema constraints.
        """
        if not user_prompt.strip():
            raise ValueError("User prompt cannot be empty.")

        self._prefix_validity_cache.clear()

        selection_prompt = self._build_selection_prompt(user_prompt)

        function_trie = FunctionTrie(
            self.model,
            selection_prompt,
            [function.name for function in self.functions],
        )

        selected_name = function_trie.select_function(
            self.model,
            function_trie.base_ids.copy(),
        )

        selected_function = next(
            function
            for function in self.functions
            if function.name == selected_name
        )

        validator = SchemaPrefixValidator([selected_function])

        prompt = self._build_prompt(
            user_prompt,
            selected_function,
        )

        input_ids = self.model.encode(prompt)
        generated_text = ""

        for _ in range(1, self.max_new_tokens + 1):
            next_token_id, next_text, token_text = self._get_next_token(
                input_ids,
                generated_text,
                validator,
            )

            input_ids.append(next_token_id)
            generated_text = next_text

            if "}" not in token_text:
                continue

            if not generated_text.rstrip().endswith("}"):
                continue

            try:
                result = json.loads(generated_text)
            except json.JSONDecodeError:
                continue

            if not isinstance(result, dict):
                continue

            if not validator.validate_call(result):
                raise RuntimeError(
                    "Generated JSON is complete but violates the schema."
                )

            parameters = cast(dict[str, object], result["parameters"])

            result["parameters"] = self._normalize_parameters(
                parameters,
                selected_function,
            )

            return cast(dict[str, object], result)

        raise RuntimeError(
            "Maximum generation length reached. "
            f"Generated text: {generated_text}"
        )
