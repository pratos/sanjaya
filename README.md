# Sanjaya

Sanjaya is a Python library for building evidence-first RLM (Recursive Language Model) agents.

It runs a loop where the model writes Python, executes it in a sandboxed REPL, inspects results, calls tools, and iterates until it can return a grounded answer.

## Why the name?

In the *Mahabharata*, Sanjaya narrates events to a blind king with "divine sight." This project follows the same idea: observe evidence and report what happened.

## Install

```bash
pip install sanjaya
```

With extras:

```bash
pip install "sanjaya[video]"      # video analysis
pip install "sanjaya[image]"      # image analysis (Pillow, HEIC, SVG)
pip install "sanjaya[document]"   # PDF/EPUB/PPTX/Markdown/text parsing
pip install "sanjaya[tracing]"    # tracing integrations
pip install "sanjaya[all]"        # everything
```

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

## Image analysis

```python
from sanjaya import Agent

agent = Agent(max_iterations=10)

answer = agent.ask(
    "What text is visible in this screenshot?",
    image="/path/to/screenshot.png",
)

print(answer.text)
print(answer.evidence)
```

Multiple images:

```python
answer = agent.ask(
    "Compare these two charts and summarize differences.",
    image=["/path/to/chart_q1.png", "/path/to/chart_q2.png"],
)
```

Supported formats: JPEG, PNG, WebP, GIF, TIFF, BMP, HEIC (requires `sanjaya[image]`), SVG (requires `sanjaya[image]`).

When `image=...` is provided, `ImageToolkit` is auto-registered if needed.

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

## Advanced Usage

### Custom model configuration

Configure different models for orchestration, sub-queries, and vision:

```python
from sanjaya import Agent

agent = Agent(
    model="openrouter:anthropic/claude-sonnet-4",  # Main orchestrator
    sub_model="openrouter:openai/gpt-4.1-mini",     # For llm_query() calls
    vision_model="openrouter:openai/gpt-4.1",      # For vision_query() calls
    caption_model="moondream:moondream3-preview",  # Cheap frame captioning
    fallback_model="openrouter:google/gemini-2.5-flash",  # Fallback on errors
    critic_model="openrouter:qwen/qwen3-30b-a3b-thinking-2507",  # Answer critic
)
```

### Budget and iteration limits

Control costs and execution time:

```python
agent = Agent(
    max_iterations=12,        # Max REPL loop iterations
    max_budget_usd=0.50,      # Stop if cost exceeds this
    max_timeout_s=120.0,      # Stop after 2 minutes
    compaction_threshold=0.85,  # Compact context at 85% token budget
    critic_threshold=70,      # Re-run if critic scores below 70/100
)

answer = agent.ask("...")
print(f"Cost: ${answer.cost_usd:.4f}")
print(f"Tokens: {answer.input_tokens} in / {answer.output_tokens} out")
print(f"Wall time: {answer.wall_time_s}s")
```

### Custom prompts with PromptConfig

Override default strategy prompts for specialized use cases:

```python
from sanjaya import Agent, PromptConfig

config = PromptConfig(
    video_strategy="Focus on timestamps and speaker identification...",
    document_strategy="Extract only numerical data and citations...",
    image_strategy="Prioritize OCR and text extraction...",
    critic="Score strictly: 0 for any hallucination, 100 for fully grounded...",
)

agent = Agent(prompts=config)
```

Load prompts from a YAML file:

```python
config = PromptConfig.from_yaml("./prompts.yaml")
agent = Agent(prompts=config)
```

Provide a pre-defined answer schema (skips schema generation LLM call):

```python
config = PromptConfig(
    answer_schema={
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "speakers": {"type": "array", "items": {"type": "string"}},
            "key_timestamps": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary"],
    }
)
```

### Custom toolkits

Create domain-specific tools:

```python
from sanjaya import Agent, Toolkit, tool

class DatabaseToolkit(Toolkit):
    def __init__(self, connection_string: str):
        self.conn_str = connection_string
        self._conn = None

    def setup(self, context: dict):
        import sqlite3
        self._conn = sqlite3.connect(self.conn_str)

    def teardown(self):
        if self._conn:
            self._conn.close()

    def tools(self):
        return [self._make_query_tool()]

    def _make_query_tool(self):
        toolkit = self

        @tool
        def run_sql(query: str) -> list[dict]:
            """Execute a SQL query and return results as a list of dicts.

            Args:
                query: SQL query to execute (SELECT only).
            """
            if not query.strip().upper().startswith("SELECT"):
                return [{"error": "Only SELECT queries allowed"}]
            cursor = toolkit._conn.execute(query)
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

        return run_sql

    def prompt_section(self) -> str:
        return """
## Database Tools
- **run_sql(query)** — Execute SELECT queries against the database.
"""

agent = Agent().use(DatabaseToolkit("./data.db"))
answer = agent.ask("What are the top 5 customers by revenue?")
```

### Composing multiple toolkits

```python
from sanjaya import Agent
from sanjaya.tools.video import VideoToolkit
from sanjaya.tools.document import DocumentToolkit
from sanjaya.tools.report import ReportToolkit

agent = Agent(max_iterations=15).use(
    VideoToolkit(max_frames_per_clip=12),
    DocumentToolkit(),
    ReportToolkit(output_dir="./reports"),
)

answer = agent.ask(
    "Compare the claims in the video with the supporting document",
    video="/path/to/presentation.mp4",
    document="/path/to/whitepaper.pdf",
)
```

### Report generation

Save analysis outputs for later retrieval:

```python
from sanjaya import Agent
from sanjaya.tools.report import ReportToolkit

agent = Agent().use(ReportToolkit(output_dir="./sanjaya_reports"))

# The agent can now call save_note(), save_qmd(), save_data() tools
answer = agent.ask(
    "Analyze this video and save a summary report",
    video="/path/to/video.mp4",
)
```

### Working with traces

Load and inspect past runs:

```python
from sanjaya import load_traces, load_trace

# Load the 5 most recent traces
traces = load_traces(n=5, artifacts_dir="./sanjaya_artifacts")

for trace in traces:
    print(f"Run: {trace['run_id']}")
    print(f"Q: {trace['question']}")
    print(f"A: {trace['answer'][:100]}...")
    print(f"Cost: ${trace['cost']['total_cost_usd']:.4f}")
    print("---")

# Load a specific trace by run_id
trace = load_trace("20250415_143022", artifacts_dir="./sanjaya_artifacts")
if trace:
    # Access detailed events
    events = trace.get("events", [])
    llm_calls = [e for e in events if "llm_call" in e.get("kind", "")]
    print(f"Total LLM calls: {len(llm_calls)}")
```

### Recursive agent queries

Enable nested RLM calls for complex decomposition:

```python
agent = Agent(
    max_iterations=10,
    max_depth=2,  # Allow one level of recursive rlm_query() calls
)

# The orchestrator can now call rlm_query(prompt) to spawn child loops
# Children inherit tools but get their own REPL and iteration budget
answer = agent.ask(
    "Analyze each speaker's arguments separately, then synthesize",
    video="/path/to/debate.mp4",
)
```

### Using with pydantic-ai providers

Pass a pre-configured provider for custom endpoints:

```python
from pydantic_ai.providers.openai import OpenAIProvider
from sanjaya import Agent

# Custom OpenAI-compatible endpoint
provider = OpenAIProvider(
    base_url="https://my-proxy.example.com/v1",
    api_key="my-api-key",
)

agent = Agent(
    model="gpt-4o",
    provider=provider,
)
```

### Accessing the Answer object

The `Answer` object provides structured access to results:

```python
answer = agent.ask("...", video="/path/to/video.mp4")

# Core answer
print(answer.text)           # Final answer string
print(answer.data)           # Structured data (if schema was used)

# Evidence trail
for ev in answer.evidence:
    print(f"Source: {ev.source}")
    print(f"Rationale: {ev.rationale}")
    print(f"Artifacts: {ev.artifacts}")

# Metrics
print(f"Iterations: {answer.iterations}")
print(f"Cost: ${answer.cost_usd:.4f}")
print(f"Input tokens: {answer.input_tokens}")
print(f"Output tokens: {answer.output_tokens}")
print(f"Wall time: {answer.wall_time_s}s")
```

### Prompt optimization with GEPA

Use [GEPA](https://github.com/gepa-ai/gepa) to evolve prompts against a benchmark:

```python
from gepa.optimize_anything import optimize_anything, GEPAConfig
from sanjaya import Agent
from sanjaya.prompts import PromptConfig

def evaluate(candidate: dict, example: dict) -> tuple[float, dict]:
    config = PromptConfig.from_dict(candidate)
    agent = Agent(prompts=config, max_iterations=6)
    answer = agent.ask(example["question"], video=example["video"])
    score = 1.0 if example["expected"] in answer.text else 0.0
    return score, {"answer": answer.text}

seed = PromptConfig(
    video_strategy="Focus on visual evidence...",
    critic="Score strictly on accuracy...",
).to_dict()

result = optimize_anything(
    seed_candidate=seed,
    evaluator=evaluate,
    dataset=train_examples,
    objective="Maximize accuracy while minimizing cost",
)

optimized = PromptConfig.from_dict(result.best_candidate)
```

### Tracing with Logfire

Enable observability with [Logfire](https://pydantic.dev/logfire):

```python
# Set LOGFIRE_TOKEN in your environment, then:
agent = Agent(tracing=True)  # tracing=True is the default

# All spans are automatically sent to Logfire:
# - sanjaya.completion (top-level)
# - sanjaya.iteration (each loop)
# - sanjaya.root_llm_call (orchestrator)
# - sanjaya.sub_llm_call.* (tool LLM calls)
# - sanjaya.tool_call (tool invocations)
```

### Resetting agent state

Clear accumulated state between runs:

```python
agent = Agent()

answer1 = agent.ask("First question", video="/path/to/v1.mp4")
print(f"Cost so far: ${agent.cost_so_far:.4f}")

# Reset clears budget, history, and workspace
agent.reset()

answer2 = agent.ask("Second question", video="/path/to/v2.mp4")
print(f"Cost after reset: ${agent.cost_so_far:.4f}")  # Only includes answer2
```

## Status

Sanjaya is actively developed. Core video + document flows are working, and evaluation/dashboard tooling is still evolving.
