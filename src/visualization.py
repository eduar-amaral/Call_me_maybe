import json


def visualize_generation_result(
    prompt: str,
    result: dict[str, object],
    elapsed_time: float,
) -> None:
    """Display the final function call and generation statistics."""

    parameters = json.dumps(
        result["parameters"],
        ensure_ascii=False,
    )

    print(f"\nPrompt:     {prompt}")
    print(f"Function:   {result['name']}")
    print(f"Parameters: {parameters}")
    print(f"Time:       {elapsed_time:.2f}s")
    print("Status:     SUCCESS")
