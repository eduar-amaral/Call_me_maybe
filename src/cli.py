
import argparse


def build_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser.

    Returns:
        Configured ``ArgumentParser`` for CLI options.
    """

    parser = argparse.ArgumentParser(
        description="LLM-based function calling with constrained decoding."
    )

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file.",
    )

    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file.",
    )

    parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
        help="Path where the results JSON file will be saved.",
    )

    return parser


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the main entrypoint.

    Returns:
        Parsed command-line namespace.
    """

    parser = build_parser()
    return parser.parse_args()
