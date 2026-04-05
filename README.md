# Sanjaya

Sanjaya is an extensible RLM (Recursive Language Model) agent framework with first-class video understanding. The agent writes Python in a sandboxed REPL, executes it, reads the result, and iterates until it can answer.

## Project status

This project is currently vibe-researched and vibe-coded. The core flows work, but validation is still in progress. It still needs stronger benchmark coverage, especially on long-video evaluations such as LongVideoBench and related datasets.

## Quick start

```bash
uv sync
cp .env.example .env
# set OPENROUTER_API_KEY (and optionally OPENAI_API_KEY / LOGFIRE_TOKEN)
```

```python
from sanjaya import Agent

agent = Agent(max_iterations=8)
text_answer = agent.ask("What is the capital of France?")
video_answer = agent.ask(
    "How many people are talking and what are they discussing?",
    video="data/longvideobench/videos/7F9IrtSHmc0.mp4",
)

print(text_answer.text)
print(video_answer.text)
```

For custom tools, register functions with `@tool` and pass them via `agent.use(...)`.

## Local app

`just dev` is currently broken after the refactor. The API/UI integration still needs to be updated before local dashboard runs are reliable.

## Documentation

See `docs/usage-examples.md`, `docs/provider-configuration.md`, `docs/architecture-duplication-audit.md`, and `docs/rlm-comparison-and-takeaways.md`.
