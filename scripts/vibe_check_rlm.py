"""Demo: Needle-in-Haystack test using sanjaya RLM."""

import argparse
import random
import string
import sys

sys.path.insert(0, "src")


def generate_massive_context(
    num_lines: int = 1_000_000,
    answer: str = None,
    position: float | None = None,
) -> tuple[str, str]:
    """Generate a large text context with a hidden 'needle'.

    Args:
        num_lines: Number of random text lines to generate
        answer: The magic number to hide (default: random 7-digit)
        position: Where to place needle (0.0-1.0). None = random 40-60%

    Returns:
        Tuple of (context string, answer string)
    """
    if answer is None:
        answer = "".join(random.choices(string.digits, k=7))

    print(f"Generating massive context with {num_lines:,} lines...")

    words = ["blah", "random", "text", "data", "content", "information", "sample"]
    lines = []

    for _ in range(num_lines):
        num_words = random.randint(3, 8)
        line = " ".join(random.choices(words, k=num_words))
        lines.append(line)

    # Insert the needle at specified position or random 40-60%
    if position is not None:
        magic_position = int(num_lines * position)
    else:
        magic_position = random.randint(int(num_lines * 0.4), int(num_lines * 0.6))

    magic_position = max(0, min(magic_position, num_lines - 1))
    lines[magic_position] = f"The magic number is {answer}"
    print(f"Magic number inserted at line {magic_position:,} ({magic_position / num_lines:.1%}) | Expected: {answer}")

    return "\n".join(lines), answer


def main():
    """Run the needle-in-haystack demo."""
    parser = argparse.ArgumentParser(description="Needle-in-Haystack test using sanjaya RLM")
    parser.add_argument("--lines", type=int, default=10_000, help="Number of lines in context")
    parser.add_argument("--answer", type=str, default=None, help="Specific magic number to use")
    parser.add_argument(
        "--position",
        type=float,
        default=None,
        help="Needle position (0.0-1.0, e.g. 0.1 = 10%% through). Default: random 40-60%%",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum number of RLM iterations",
    )
    parser.add_argument(
        "--trace-demo",
        action="store_true",
        help="Run a deterministic, small context preset for tracing",
    )
    args = parser.parse_args()

    if args.trace_demo:
        random.seed(42)
        if args.lines == 10_000:
            args.lines = 250
        if args.answer is None:
            args.answer = "4242424"
        if args.position is None:
            args.position = 0.5
        if args.max_iterations == 10:
            args.max_iterations = 5
        print("Running deterministic trace demo preset")

    from sanjaya.rlm_repl import RLM_REPL

    # Generate context
    context, answer = generate_massive_context(
        num_lines=args.lines,
        answer=args.answer,
        position=args.position,
    )
    query = "I'm looking for a magic number. What is it?"

    # Create RLM instance
    rlm = RLM_REPL(
        model="openrouter:openai/gpt-5.3-codex",
        recursive_model="openrouter:openai/gpt-4.1-mini",
        enable_logging=True,
        max_iterations=args.max_iterations,
    )

    # Run completion
    result = rlm.completion(context=context, query=query)

    print(f"\nResult: {result}. Expected: {answer}")


if __name__ == "__main__":
    main()
