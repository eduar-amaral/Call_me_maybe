
*This project has been created as part of the 42 curriculum by eamaral-.*

# call_me_maybe

## Description

This project explores function calling with small language models (SLMs). The goal is to transform natural-language requests into valid JSON function calls while respecting the available function definitions and their parameter requirements.

The system uses a language model to select the appropriate function and extract its arguments. Constrained decoding ensures that generated tokens remain compatible with the expected JSON structure and function definitions.

## Instructions

### Requirements

- Python 3.10 or later
- uv installed and available on PATH

### Setup

Install the project dependencies:

```bash
uv sync
```

### Common commands

```bash
make run
make lint
make lint-strict
make clean
make fclean
```

### Running the project

Run with the default input and output paths:

```bash
uv run python -m src
```

Or specify the paths explicitly:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

The application loads the function definitions and input prompts, generates the corresponding function calls, and writes the results to the output file.

## Algorithm explanation

The constrained-decoding pipeline follows a token-by-token approach:

1. Available function definitions establish the expected function-call structure.
2. A prompt provides the language model with the available functions and the user's request.
3. The model produces logits representing possible next tokens.
4. Candidate tokens are considered in descending order of their logits.
5. Each candidate is evaluated against the current partial JSON sequence.
6. A token is accepted only if the resulting prefix remains compatible with a valid function call.
7. Generation stops when a complete function call has been produced and validated.

The language model remains responsible for selecting the function and generating its parameter values. The constrained-decoding mechanism restricts the output so that invalid JSON structures, unknown function names, and incompatible parameters are rejected during generation.

## Design decisions

- **Constrained decoding:** Candidate tokens are filtered before acceptance, keeping generation compatible with the expected function-call structure.
- **Prefix validation:** Partial JSON sequences are validated before accepting new tokens.
- **Prefix caching:** Previously evaluated prefixes can reuse validation results, avoiding unnecessary repeated work.
- **Schema-aware validation:** Function names, parameter names, and parameter types are checked against the available function definitions.
- **Separation of concerns:** Prompt construction, token selection, JSON parsing, and function-call validation are handled by separate components.

## Performance analysis

The main performance bottleneck is the repeated forward pass through the language model.

Constrained decoding requires evaluating candidate tokens at each generation step. This introduces additional computation compared with unrestricted generation. The implementation reduces avoidable overhead through prompt construction and prefix-validation caching, but model inference remains the dominant cost.

During a local execution using the 11 prompts provided in `function_calling_tests.json`, all 11 function calls completed successfully.

Observed execution times:

- Total execution time: approximately 7 minutes and 45 seconds.
- Fastest prompt: approximately 25 seconds.
- Slowest prompt: approximately 82 seconds.

These measurements depend on the hardware and execution environment. All 11 prompts completed in less than five minutes individually during this run.

## Challenges faced

The main challenge was balancing the model's natural generation behavior with strict structural requirements.

A language model can produce tokens that appear reasonable in natural language but are invalid in the context of a partially generated JSON object. The constrained-decoding approach addresses this by evaluating candidate prefixes before accepting them.

Another challenge was working within the project restrictions. The implementation uses the provided SDK through its public interface and avoids relying on private SDK methods or prohibited libraries.

## Validation

The project was validated through runtime execution and static analysis.

### Runtime validation

The application was executed against the 11 prompts in:

```text
data/input/function_calling_tests.json
```

All 11 prompts produced successful function calls with the expected function names and parameter values.

### Static analysis

The following commands completed successfully:

```bash
make lint
make lint-strict
```

These commands run Flake8 and Mypy, including Mypy's strict checks.

## Example usage

Input:

```text
What is the sum of 2 and 3?
```

Generated function call:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2.0,
    "b": 3.0
  }
}
```

The generated result follows the expected function-call structure and uses the parameter definitions supplied to the application.

## Resources

- Qwen/Qwen3-0.6B model documentation
- Hugging Face documentation on language models and logits
- Python JSON documentation
- Pydantic documentation
- NumPy documentation

### AI usage

AI tools were used to:

- Explore and understand constrained decoding and prefix validation.
- Review code structure and explain the generation pipeline.
- Discuss possible optimizations while preserving the assignment requirements.
- Assist with documentation and debugging.

Implementation decisions, code changes, and validation results were reviewed throughout the development process.

## Notes

This project focuses on combining language-model function selection with constrained generation.

The implementation prioritizes valid, schema-compatible function calls while respecting the assignment's technical restrictions and using the provided SDK interface.
