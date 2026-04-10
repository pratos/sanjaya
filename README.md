# Sanjaya

Sanjaya is an extensible RLM (Recursive Language Model) inspired agent framework with first-class video understanding. The agent writes Python in a sandboxed REPL, executes it, reads the result, and iterates until it can answer.

It answers questions by writing Python inside a sandboxed REPL, executing the code, reading the results, and iterating until it can produce a grounded answer.

## Why the name?

In the *Mahabharata*, Sanjaya narrates events to a blind king with "divine sight." The project name reflects the same goal: help a model observe evidence (video + transcript) and report what happened.

## Current status

This project is currently vibe-researched and vibe-coded. The core flows work, but validation is still in progress. It still needs stronger benchmark coverage, especially on long-video evaluations such as LongVideoBench and related datasets.

Pydantic Monty's philosophy about sandboxing has heavily influenced the true RLM parts: direct context variable access and a full stdlib access. In the project blog:

Monty is a Python interpreter written in Rust. Not CPython-with-restrictions. Not Python compiled to WASM. A from-scratch bytecode VM that uses Ruff's parser to turn Python source into its own bytecode format.

What it supports:

- Functions (sync and async), closures, comprehensions
- f-strings, type hints, dataclasses when defined on the host
- sys, typing, asyncio, pathlib standard library modules. re, datetime, json coming soon
- External function calls — the mechanism for interacting with the host
- Snapshotting — serialize execution state mid-flight to bytes, resume later or elsewhere
- Type checking — ships with ty bundled in the binary
- Memory, recursion and execution time limits within the interpreter
- REPL support - from our testing LLMs strongly assume a REPL - that functions and values it previously defined are available when code is next executed

What it doesn't support:

- Classes - coming soon
- Match statements - coming soon
- context managers - coming soon
- Full standard library - we'll add more over time as and when the LLM wants to use it
- Third-party packages - Monty will probably never support 3rd party libraries.
It sits between implementations of RLM loop & ReACT agent.


## Install

```bash
uv sync
cp .env.example .env
# set OPENROUTER_API_KEY
# optional: OPENAI_API_KEY, LOGFIRE_TOKEN, MOONDREAM_API_KEY
```

## Quick start

```python
from sanjaya import Agent

agent = Agent(max_iterations=8)

text_answer = agent.ask("What is the capital of France?")
print(text_answer.text)

video_answer = agent.ask(
    "How many people are talking and what are they discussing?",
    video="data/longvideobench/videos/7F9IrtSHmc0.mp4",
)
print(video_answer.text)
```

`Agent.ask()` returns an `Answer` object with:
- `text`
- `evidence`
- `iterations`
- token/cost fields (`input_tokens`, `output_tokens`, `cost_usd`)
- `wall_time_s`

## How video analysis works

When `video=...` is provided, Sanjaya auto-registers `VideoToolkit` and uses tools such as:

- `list_windows()`
- `extract_clip()`
- `sample_frames()`
- `caption_frames()`
- `vision_query()` / `vision_query_batched()`
- `search_transcript()`

The loop encourages evidence-first behavior: explore clips/frames first, then call `done()` in a dedicated final step.

## Custom tools

Register custom tools with `@tool` and `agent.use(...)`:

```python
from sanjaya import Agent, Toolkit, tool

class MyToolkit(Toolkit):
    @tool
    def lookup(self, query: str) -> str:
        """Lookup internal information."""
        return f"result for: {query}"

agent = Agent().use(MyToolkit())
answer = agent.ask("Find the latest status update")
```

## Local dashboard/dev workflow

From project root:

- `just dev` — run API + UI via Overmind
- `just api` — run FastAPI backend (`:8000`)
- `just ui` — run Next.js frontend (`:5100`)
- `just demo` — run bundled benchmark/demo prompts
