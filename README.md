# Sanjaya

Sanjaya is a Python library for building evidence-first RLM (Recursive Language Model) agents.

It runs a loop where the model writes Python, executes it in a sandboxed REPL, inspects results, calls tools, and iterates until it can return a grounded answer.

## Why the name?

In the *Mahabharata*, Sanjaya narrates events to a blind king with "divine sight." This project follows the same idea: observe evidence and report what happened.

## Install

### Base package

```bash
uv add sanjaya
# or
pip install sanjaya
```

### With extras

```bash
uv add "sanjaya[video]"
uv add "sanjaya[document]"
uv add "sanjaya[all]"

# or
pip install "sanjaya[video]"
pip install "sanjaya[document]"
pip install "sanjaya[all]"
```

Extras:

- `video` — video analysis dependencies
- `document` — PDF/EPUB/PPTX/Markdown/text document parsing dependencies
- `tracing` — tracing integrations
- `all` — all optional features

## Configuration

Set at least:

- `OPENROUTER_API_KEY`

Optional (depends on your model/provider setup):

- `OPENAI_API_KEY`
- `MOONDREAM_API_KEY`
- `LOGFIRE_TOKEN`

## Quick start

```python
from sanjaya import Agent

agent = Agent(max_iterations=8)

answer = agent.ask("What is the capital of France?")
print(answer.text)
```

## Video analysis

```python
from sanjaya import Agent

agent = Agent(max_iterations=12)

answer = agent.ask(
    "How many people are speaking and what are they discussing?",
    video="/path/to/video.mp4",
    # subtitle="/path/to/subtitle.json",  # optional
)

print(answer.text)
print(answer.evidence)
```

When `video=...` is provided, `VideoToolkit` is auto-registered if you have not already added one.

## Document analysis

```python
from sanjaya import Agent

agent = Agent(max_iterations=12)

answer = agent.ask(
    "Summarize the key claims and supporting evidence.",
    document=["/path/to/report.pdf", "/path/to/appendix.md"],
)

print(answer.text)
print(answer.evidence)
```

Supported document types include `.pdf`, `.epub`, `.pptx/.ppt`, `.md`, and `.txt`.

When `document=...` is provided, `DocumentToolkit` is auto-registered if needed.

## Custom tools

```python
from sanjaya import Agent, Toolkit, tool

class MyToolkit(Toolkit):
    @tool
    def lookup(self, query: str) -> str:
        return f"result for: {query}"

agent = Agent().use(MyToolkit())
answer = agent.ask("Find the latest status update")
print(answer.text)
```

## Answer object

`Agent.ask()` returns an `Answer` object with:

- `text`
- `evidence`
- `iterations`
- `input_tokens`, `output_tokens`, `cost_usd`
- `wall_time_s`

## Development (repo)

If you are working from the repository:

- `just dev` — API + UI
- `just api` — FastAPI backend (`:8000`)
- `just ui` — Next.js frontend (`:5100`)
- `just demo` — run demo prompts

## Status

Sanjaya is actively developed. Core video + document flows are working, and evaluation/dashboard tooling is still evolving.
