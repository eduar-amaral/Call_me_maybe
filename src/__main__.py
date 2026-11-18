import json
import sys
from pathlib import Path
from time import perf_counter

from src.cli import parse_args
from src.generation import ConstrainedGenerator
from src.llm import LanguageModel
from src.parser import load_function_definitions, load_prompts
from src.visualization import visualize_generation_result


def main() -> None:
    """Run constrained generation for all prompts and save output.

    Raises:
        SystemExit: Exits with status code 1 on handled runtime/input errors.
    """

    args = parse_args()

    try:
        prompts = load_prompts(args.input)
        functions = load_function_definitions(args.functions_definition)
        generator = ConstrainedGenerator(
            model=LanguageModel(),
            functions=functions,
        )

        results: list[dict[str, object]] = []

        for index, prompt in enumerate(prompts, start=1):
            print(f"\n[{index}/{len(prompts)}] Generating...")

            start_time = perf_counter()

            result = generator.generate(prompt.prompt)

            elapsed_time = perf_counter() - start_time

            visualize_generation_result(
                prompt.prompt,
                result,
                elapsed_time,
            )

            results.append(
                {
                    "prompt": prompt.prompt,
                    "name": result["name"],
                    "parameters": result["parameters"],
                }
            )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(results, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")

        print(f"Generated {len(results)} function calls")
        print(f"Saved results to {output_path}")

    except KeyboardInterrupt:
        print("\nGeneration interrupted by user.")
        raise SystemExit(130)

    except (
        FileNotFoundError, PermissionError, ValueError, RuntimeError, OSError
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
