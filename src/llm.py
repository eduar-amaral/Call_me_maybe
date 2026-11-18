
import importlib
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, PrivateAttr


class LanguageModel(BaseModel):
    """Interface wrapper around the provided ``llm_sdk`` model."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    _model: Any = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        """Instantiate the SDK model implementation.

        Args:
            context: Pydantic initialization context.
        """
        sdk_module = importlib.import_module("llm_sdk")
        self._model = sdk_module.Small_LLM_Model()

    def encode(self, text: str) -> list[int]:
        """Convert text into token ids.

        Args:
            text: Input text prompt.

        Returns:
            Token ids representing the text.
        """
        encoded = self._model.encode(text)
        return cast(list[int], encoded[0].tolist())

    def decode(self, token_ids: list[int]) -> str:
        """Convert token ids back into text.

        Args:
            token_ids: Token sequence to decode.

        Returns:
            Decoded text string.
        """
        return cast(str, self._model.decode(token_ids))

    def get_logits(self, input_ids: list[int]) -> list[float]:
        """Get next-token logits for an input prefix.

        Args:
            input_ids: Current prompt token ids.

        Returns:
            Logit scores for each vocabulary token.
        """
        return cast(
            list[float],
            self._model.get_logits_from_input_ids(input_ids),
        )

    def get_vocab_path(self) -> str:
        """Get path to the model vocabulary file.

        Returns:
            Filesystem path to the tokenizer vocabulary file.
        """
        return cast(str, self._model.get_path_to_vocab_file())

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
    ) -> str:
        """Generate text with greedy decoding.

        Args:
            prompt: Prompt text used as generation context.
            max_new_tokens: Maximum number of output tokens.

        Returns:
            Decoded generated text.

        Raises:
            ValueError: If ``max_new_tokens`` is lower than 1.
        """
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

        input_ids = self.encode(prompt)
        generated_ids: list[int] = []

        for _ in range(max_new_tokens):
            logits = self.get_logits(input_ids)

            next_token_id = max(
                range(len(logits)),
                key=logits.__getitem__,
            )

            generated_ids.append(next_token_id)
            input_ids.append(next_token_id)

        return self.decode(generated_ids)
