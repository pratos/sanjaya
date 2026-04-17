---
title: "Building Sanjaya: An RLM That Programs Its Way Through Videos and Images"
description: "How I built an open-source Recursive Language Model agent for multimodal understanding, with benchmark results, cost analysis, and lessons from 11 days of building."
date: 2026-04-15
author: Prathamesh Sarang
tags: [rlm, recursive-language-model, agent, video-understanding, image-retrieval, benchmarks, python]
---

# Building Sanjaya: An RLM Agent That Programs Its Way Through Videos and Images

**[Sanjaya](https://github.com/pratos/sanjaya)** is an open-source Python library (`uv add sanjaya`) inspired by Recursive Language Model (RLM) agents, built for multimodal understanding: video, documents, and images. Instead of prompting a model to answer a question, the model writes a Python program that answers it. The program searches transcripts, extracts video clips, samples frames, queries vision models, and iterates, all inside a sandboxed Read Eval Print Loop (REPL).

A live demo runs at [sanjaya.bayalis.in](https://sanjaya.bayalis.in) if you want to poke at it without installing anything.

I ran Sanjaya on a few real workloads to see how it holds up.

On [PhotoBench](https://huggingface.co/datasets/SorrowTea/PhotoBench), a benchmark where you answer questions about a large photo album ("which photos were taken at the beach?", "making dinner in the kitchen", "nighttime pedestrian street in Guiyang"), Sanjaya surfaced 98.9% of the relevant photos on the highest-scoring album and 86–89% on the harder ones. Roughly 29 to 37 cents per query.

Then long videos: clips from LongVideoBench and a few I picked by hand. Same pattern. I asked questions like "when does the speaker first mention X?" and got useful answers for $0.05 to $1.15 per query, depending on how much the model had to look at.

Those numbers might not mean much yet. The point is: these are tasks a normal model would either refuse (too much video, too many photos to fit in context) or hallucinate its way through. An RLM handles them by writing code to *go look*, instead of trying to cram the whole album or video into a single prompt.

This post covers what I learned building it: what an RLM is and isn't, the decisions that shaped the Sanjaya API, some vibe-benchmarking.

---

## Frontier models have a context problem

One of the biggest constraints on today's frontier models (Opus, Codex, the rest) is something called context rot. From [Chroma's technical report](https://www.trychroma.com/research/context-rot#introduction):

> Large Language Models (LLMs) are typically presumed to process context uniformly—that is, the model should handle the 10,000th token just as reliably as the 100th. However, in practice, this assumption does not hold. We observe that model performance varies significantly as input length changes, even on simple tasks

Those were structured experiments. Chroma notes performance degrades further under real-world conditions. If you've used ChatGPT or any other service for a long conversation, you've probably noticed the model gets duller the more you chat, and a fresh session usually fixes it.

For longer-form tasks, like asking an agent to pour over hundreds of documents to find a relevant quote, people have tried a bunch of workarounds. RAG and its variants are the most common. But with RAG comes a whole data-engineering scaffold: chunking, embeddings, vector stores, retrieval tuning. Extra pipeline, extra cost.

## How Did Sanjaya Get Started?

The project started at the [Codex Community Hackathon](https://luma.com/kauset3d?tk=ybZFAc) in Mumbai. I'd been interested in DSPy and RLMs since reading [Prime Intellect's blog on RLMs](https://www.primeintellect.ai/blog/rlm), and the hackathon felt like a good excuse to actually build something.

The day before, I did a spike using Alex Zhang's [rlm-minimal](https://github.com/alexzhang13/rlm-minimal) as the base. I wanted a sandbox that didn't need anyone to set up infra or plug in an API key, so I pulled in [Pydantic's Monty](https://github.com/pydantic/monty). Having committed to Pydantic, I also ripped out the OpenAI core and swapped in pydantic-ai and logfire for tracing.

The original plan was a prose editor built on top of an RLM. But then, I remembered the Prime Intellect post mentioning multi-modal RLMs. And that's where RLMs for video idea came in.

Here's the problem with video. A 30-minute lecture is roughly 5,000–7,000 tokens of transcript depending on format, plus whatever the visuals add. If you stuff 6K tokens into every prompt and the agent takes 8 turns to answer, you've paid for ~48,000 tokens. Same transcript, re-read eight times. Add frames or vision calls and costs spiral.

And transcripts only get you so far. Imagine an editor cutting a concert video who needs every shot where the artist is on stage. Text-only search can't help. The signal lives in where the camera is pointed, which no transcript captures. Visual grounding is the only way. Gemini's models can watch video, but they cap out at a million tokens. [Twelve Labs](https://www.twelvelabs.io/product/search) has great products for this kind of search and analysis. RLMs seemed like a natural third option worth exploring.

RLMs for video also just felt like a fun thing to build at a hackathon. Risky, but worth trying.

---

## RLMs: Are they the free lunch?

The simple fix for context rot: don't put the long context in the prompt. Instead, hand the model a Python REPL and let it write code to go look.

That's what a Recursive Language Model (RLM) does. The model writes code. The code searches the input, chunks it when needed, reads only what's relevant. The model can also call itself (or a smaller model) for sub-tasks ([Zhang et al., 2025](https://arxiv.org/abs/2512.24601)).

Here's an example. Spoilers ahead for A Song of Ice and Fire. One of the running mysteries:

> Who's the prince that was promised?

The corpus is five novels, around 1.7 million words, or roughly 2.2 million tokens. Too much for a single prompt. You could build a RAG pipeline: chunker, embeddings, vector store, retrieval prompt, synthesis prompt. That works, but you pick the retrieval strategy before you know what the question needs.

With an RLM, the model writes the strategy:

[![Watch: ASOIAF document search with Sanjaya](https://img.youtube.com/vi/ZAbTV0ekqb0/hqdefault.jpg)](https://youtu.be/ZAbTV0ekqb0)

```python
# Derived from an earlier pass over the corpus
candidates = ["Daenerys Targaryen", "Jon Snow", "Stannis Baratheon", "Aegon Targaryen"]

evidence = {}
for name in candidates:
    hits = search_documents(
        f"{name} prince that was promised Azor Ahai",
        top_k=5,
    )
    passages = [read_chunk(h["doc_id"], h["chunk_index"]) for h in hits]
    evidence[name] = llm_query(
        f"From these passages, extract direct prophecy text supporting {name} "
        f"with book and chapter. Passages: {passages}"
    )

answer = llm_query(f"Synthesize a ranked answer from this evidence: {evidence}")
print(answer)
```

Two caveats.

First, the `candidates` list is hard-coded to keep the snippet short. In a real run, the model has to find these names itself first.

Second, `llm_query` passes the passages into a sub-LLM. That's fine. The rule is that the *main* model never reads the full text. Sub-calls get a fresh context, do one job, and go away.

The paper names three properties that make this work. In plain terms:

1. **The text stays in the REPL.** The five novels aren't in the prompt. They're indexed, and the model reads them through code: `search_documents()` for keyword search, `read_chunk()` for a specific chunk, `list_documents()` to see what's there. The main model only ever sees the passages it cites in the final answer. Everything else was looked at and dropped along the way.

2. **Variables stick around between turns.** The REPL is one long session. The model builds a `candidates` list, adds to an `evidence` dict, and updates earlier entries when new quotes turn up. When Melisandre's "only Snow" line shows up in a later search, it gets appended to Jon's entry. Daenerys and Stannis stay as they were.

3. **The model can call itself.** `llm_query()` runs one prompt against a smaller model. `rlm_query()` spawns a whole child RLM (its own REPL, its own tools, its own loop) for sub-problems that need their own exploration. For four candidates, the model can fan out: one child per name, running in parallel, each digging into that character's prophecies. Then a final pass pulls the answers together.

Sanjaya implements all three properties: symbolic handles (tools like `search_documents`, `search_transcript`), persistent state (a MontyRepl session that carries variables across iterations), and recursion (`llm_query` and `rlm_query`). Since Zhang et al.'s paper landed, DSPy has shipped `dspy.RLM` (3.1.3, Feb 2026) and Trampoline AI has shipped `predict-rlm` (Apr 2026). Both are general-purpose; Sanjaya is specialized for multimodal retrieval.

So, free lunch? Not really.

For simple questions, the RLM loop adds overhead. The model is writing code when one prompt would have done.

It pays off on the hard stuff: long inputs, multi-hop retrieval, questions where you don't know up front which part of the input matters. Cost scales with how hard the question is; input length stops being the main axis.

For long video, photo albums (needle-in-haystack problems), and large document sets, that trade seems worth it.


---

## Early results: where it actually lands

Not a full evaluation. I had a budget and spent it on whatever seemed most informative. Here's what the numbers say so far.

### Setup

GPT-5.3-Codex as the orchestrator. GPT-4.1-mini for vision across both benchmarks, plus Moondream-2B as an additional video-frame captioner. Qwen3-30b as the critic. 10-iteration cap. $5 budget per query.

### PhotoBench (images)

[PhotoBench](https://huggingface.co/datasets/SorrowTea/PhotoBench) asks natural-language questions over personal photo albums, like "light yellow patchwork plush pajamas selfie" or "sunset photos from the beach trip." You return the right images from albums of 58–69 photos. I ran 20 queries per album across three albums (60 total).

| Album   | Recall    | Precision | F1        | $/Query   |
| ------- | --------- | --------- | --------- | --------- |
| 1 (64 images)   | 98.9%     | 80.0%     | 83.1%     | $0.29     |
| 2 (58 images)   | 89.2%     | 53.3%     | 60.9%     | $0.34     |
| 3 (69 images)   | 86.3%     | 60.9%     | 64.9%     | $0.37     |
| **All**         | **91.5%** | **64.7%** | **69.6%** | **$0.33** |

MRR stays above 75% across all three albums (92.1%, 75.1%, 88.9%).


A full run across all 60 queries:

[![Watch: PhotoBench run across 60 queries](https://img.youtube.com/vi/iMpz8LmMSD0/hqdefault.jpg)](https://youtu.be/iMpz8LmMSD0)

**When it works.** Album 1 hit 98.9% recall at 80% precision. Recall stays 86–99% across all three albums because the pipeline is inclusive by design (caption search first, vision verification after, borderline matches kept). MRR stays above 75% everywhere, so even when precision slips, the right image usually lands near the top.

**When it breaks.** Precision drops to 53–61% on Albums 2 and 3 whenever a query needs data the toolkit doesn't have: GPS for "photos from Paris," face recognition for "photos with Sarah," calendar data for "photos from last Christmas." The model tries to infer these from pixels and over-includes.

**Takeaway.** Strong recall, decent precision. When it fails, it fails by over-including rather than silently missing the target. Anything needing metadata the tools don't expose is a write-off.

### Video

No ground truth, no automated scoring. Just demo prompts and LongVideoBench clips across v2–v5 of the codebase. I'm watching behavior and cost here; benchmark accuracy comes later.

**When it works.**

A K-pop performance from LongVideoBench. No usable transcript, audio cues aren't enough. Asked: "Give me the summary and how many dancers are present in the video." Sanjaya worked through the frames and got it.

[![Watch: K-pop dancer-count answer](https://img.youtube.com/vi/StZpE6nlYM0/hqdefault.jpg)](https://youtu.be/StZpE6nlYM0)

Another LongVideoBench question: pick the correct sequence of scenes. Ground truth was Option A. The correct order: military-hat man → dark-blue-uniform man with white flower → camera man by yellow rock. Sanjaya's answer:

> "[Option A] A man wearing a military hat, a long coat, and long boots, with white and red English words in the frame → then a man in a dark blue military uniform with a white flower/medals on the right and a yellow paper sheet on the left → then the man in a white short-sleeve shirt holding a camera by a yellow rock under a blue sky."

Correct sequence, correct option. The benchmark scorer still marked this wrong. The MCQ extractor expected a verbatim option string; Sanjaya gave it a paraphrase with arrows.

[![Watch: LongVideoBench scene-sequence answer](https://img.youtube.com/vi/gyoHQqRqVvw/hqdefault.jpg)](https://youtu.be/gyoHQqRqVvw)

An MKBHD product review. Sanjaya extracted structured features with timestamps and visual evidence.

[![Watch: MKBHD product-review analysis](https://img.youtube.com/vi/tERmHnFV-14/hqdefault.jpg)](https://youtu.be/tERmHnFV-14)

**When it breaks.**

"Find every 3-pointer in this Steph Curry game." Five runs across versions returned counts of 9, 9, 19, 21, and 26. The actual count is ~114 (discounting replays). Every run under-reports by a large margin. The model sampled the transcript and a subset of frames, and never did the exhaustive frame-by-frame pass the question actually needed.

[![Watch: Curry 3-pointer count, under-reported](https://img.youtube.com/vi/xqs-tvMzCXc/hqdefault.jpg)](https://youtu.be/xqs-tvMzCXc)

Prompt regressions are also real. On one podcast question, v5's base prompt produced a worse answer than v4. Prompt changes at this layer are still brittle.

[![Watch: v4→v5 podcast regression](https://img.youtube.com/vi/D9gXObHc1EE/hqdefault.jpg)](https://youtu.be/D9gXObHc1EE)

**Takeaway.** Cost depends more on the question than the video. Cheap ones are whatever the transcript can answer. Expensive ones push the model onto frames. And the ones Sanjaya just gets wrong are the counting questions (every 3-pointer, every ad break), because sampling doesn't catch everything.

### Cost shape vs. Gemini and Twelve Labs

Numbers below are from public pricing pages as of April 2026. I haven't run the same prompts through all three yet; that's the next iteration. Treat this as napkin math rather than a benchmark.

Same workload: one question on a 50-minute video.

| System                        | Per-query cost (approx.)          |
| ----------------------------- | --------------------------------- |
| Gemini 2.5 Flash              | ~$0.24                            |
| Gemini 3.1 Flash-Lite Preview | ~$0.20                            |
| Twelve Labs (Developer)       | ~$2.18 first query, ~$0.004 after |
| Sanjaya (measured)            | $0.05–$0.85                       |

Where the numbers come from. Gemini 2.5 Flash charges $0.30 per 1M video tokens (~790K tokens for 50 minutes at roughly 263 tokens/second). Twelve Labs Developer tier is $0.042/min indexing + $0.0015/min infrastructure ≈ $2.18 for a 50-min video, then $4 per 1,000 search queries ($0.004 each). Sanjaya numbers are direct measurements from the runs above.

Three different shapes worth calling out:

- **Gemini** is cheapest for one-off questions on a video you won't revisit. Every question pays to re-read the whole video.
- **Twelve Labs** is cheapest if you'll ask hundreds of questions on the same indexed library. The indexing cost amortizes and queries cost fractions of a cent.
- **Sanjaya** is cheapest on simple transcript-anchored questions and most expensive on exhaustive vision work. No upfront indexing cost beyond optional transcription.

These are approximations only. Next write-up will run identical prompts through each and report real cost and answer quality side by side.

### What I haven't measured

- LongVideoBench (6,678 multiple-choice questions). 14 sample videos downloaded, no runner yet.
- Any local model. Everything ran through cloud APIs; Moondream runs through Modal.
- A real Gemini or Twelve Labs head-to-head with identical prompts.
- The PhotoBench leaderboard submission. File exists, never submitted.

---

## What I Figured Out Along the Way

Four decisions I'd make again. Most I got wrong first.

### 1. Keep the tool surface small

Every tool I expose is a strategy decision I've made for the model. Every tool I don't expose forces the model to write the strategy itself in Python.

I started with 10 tools for video and cut to 7. The ones I dropped: regex over transcript results, timestamp arithmetic, list aggregation. The model does all of those fine in Python. A tool would only limit it to whatever shape I picked. The ones I kept: `ffmpeg` clip extraction, vision API calls, subtitle parsing. Those are things the sandbox literally can't do.

The rule: wrap what the sandbox can't provide (I/O, external APIs, binaries). Everything else (loops, conditionals, aggregation, formatting) the model writes.

### 2. Make the model earn `done()`

LLMs rush. Given a video question, the model will call `done()` after reading two transcript segments, before it looks at a single frame. The answer sounds confident and has no visual grounding.

Two things fix this. First, a critic. A separate model (Qwen3-30b) scores every candidate answer 0–100 against a dynamically-generated JSON schema. Below 70, the answer gets kicked back with specific gaps:

```
Your answer was evaluated (score: 45/100).
Issues:
- No visual evidence cited for any claimed 3-pointers
- Timestamps not verified against frame analysis
- Missing jersey number identification from frames

Investigate these gaps and call done() again with an improved answer.
```

The model extracts clips, samples frames, runs vision queries, and tries again. The critic doesn't care about modality. It evaluates against the schema, which was generated from the question.

Second, two guard rules in the loop itself. `done()` in the same response as other code blocks gets suppressed (observe first, then decide). If vision tools are registered and none have been used, the first `done()` gets suppressed too. Both live in `core/loop.py`.

The critic catches bad answers. The guards catch the skip-ahead pattern before the critic ever runs. Neither is in the Zhang et al. paper. They're patches for the specific failure mode of LLMs rushing past observation.

### 3. Move long text behind a search tool

The single biggest cost drop in the whole project came from taking transcripts out of the system prompt and putting them behind a `search_transcript()` tool. Same for image captions: searched on demand, never pasted in wholesale.

When the transcript sits in the prompt, the model skims and guesses. When it has to search, it writes targeted queries. Cheaper and better.

That's the move from v2/v3 (transcript in the system prompt) to v4 (transcript behind a search tool). Simple factual questions dropped from ~$0.37 per run down to ~$0.04, because those questions stopped paying to re-read the full transcript on every turn.

### 4. Pick a sandbox that actually isolates

rlm-minimal runs code through `exec()` with a builtins allowlist. That holds up for a research demo.

For anything user-facing, it falls apart. `__import__` is allowed, so `import os; os.system(...)` works. The classic `().__class__.__bases__[0].__subclasses__()` trick reaches everything else.

If a user can type a prompt that reaches my model, they can type a prompt that writes code the model will run. The sandbox has to actually hold.

The options I weighed:

- **Pyodide / WASM via Deno.** Real isolation. But ~2,800ms cold start per run, and a JS bridge that's awkward on the server. DSPy's `dspy.RLM` uses this.
- **Docker.** Real isolation, ~195ms start, plus a daemon and images to maintain. The full rlm repo supports this (alongside Modal, E2B, Daytona).
- **pydantic-monty.** A Rust bytecode VM with its own parser (Ruff's), bytecode, and heap. Not a CPython sandbox. Because `os`, `subprocess`, and `requests` don't exist in that runtime, they can't be imported. Filesystem access is intercepted and host-controlled. Memory, time, and allocation limits enforced at the VM level. ~0.06ms start.

Monty's tradeoff is language completeness: no classes yet, limited stdlib. For the RLM use case (tool calls, control flow, string/list/dict manipulation) that's fine. The model writes Python that calls injected functions; it doesn't need `import subprocess`.

The practical payoff: Sanjaya ships with no extra infrastructure. No Deno install, no Docker daemon. `uv add sanjaya` and the sandbox works.

And because Monty exposes a `dump()`/`load()` snapshot API, agent runs can be paused mid-execution and resumed later. Useful for long video workloads where a single question might take minutes.

---

## What's next

A few things on the near-term list, in rough priority order:

1. **Head-to-head benchmarks.** The same prompts run against Gemini 2.5 Flash, Gemini 3.1 Flash-Lite, and Twelve Labs. Real numbers will replace the napkin math above.
2. **LongVideoBench runner.** The 6,678-question benchmark deserves an actual submission, not a 14-video spot check.
3. **Local models.** Moondream already runs on Modal. Next is a full local stack (local orchestrator plus local critic) so you can run Sanjaya without cloud API keys.
4. **Exhaustive-enumeration mode.** A different sampling strategy for "find every X" questions. The current loop misses too much. Probably a sweep-the-video toolkit rather than a prompt-level fix.

If you try Sanjaya and hit a rough edge, [open an issue](https://github.com/pratos/sanjaya/issues). The codebase is small enough to fix quickly. Star the repo if you want to track future write-ups.

---



## How do I install Sanjaya?

```bash
uv add sanjaya              # core
uv add "sanjaya[video]"     # video analysis (ffmpeg required)
uv add "sanjaya[image]"     # image analysis (Pillow, HEIC, SVG)
uv add "sanjaya[document]"  # PDF/EPUB/PPTX/Markdown parsing
uv add "sanjaya[all]"       # everything
```

Requires Python ≥ 3.12. Set `OPENROUTER_API_KEY` for the default model configuration.

Or skip setup entirely and try the hosted demo at [sanjaya.bayalis.in](https://sanjaya.bayalis.in).

---

## References

- Zhang, A., Kraska, T., Khattab, O. (2025). "Recursive Language Models." *arXiv:2512.24601*. [arxiv.org/abs/2512.24601](https://arxiv.org/abs/2512.24601)
- Zhang, A. Reference RLM implementation. [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm)
- Zhang, A. Minimal RLM implementation. [github.com/alexzhang13/rlm-minimal](https://github.com/alexzhang13/rlm-minimal)
- Stanford NLP. `dspy.RLM` module (3.1.3, Feb 2026). [github.com/stanfordnlp/dspy](https://github.com/stanfordnlp/dspy)
- Trampoline AI. `predict-rlm` (Apr 2026). [github.com/Trampoline-AI/predict-rlm](https://github.com/Trampoline-AI/predict-rlm)
- Chroma Research. "Context Rot." [trychroma.com/research/context-rot](https://www.trychroma.com/research/context-rot)
- pydantic-ai: Agent framework. [ai.pydantic.dev](https://ai.pydantic.dev/)
- pydantic-monty: Sandboxed Python REPL. [github.com/pydantic/pydantic-monty](https://github.com/pydantic/pydantic-monty)
- PhotoBench: Image retrieval benchmark. [huggingface.co/datasets/SorrowTea/PhotoBench](https://huggingface.co/datasets/SorrowTea/PhotoBench)
- LongVideoBench. [longvideobench.github.io](https://longvideobench.github.io/)
- Gemini API pricing. [ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing)
- Twelve Labs pricing. [twelvelabs.io/pricing](https://www.twelvelabs.io/pricing)
- Full implementation comparison against rlm-minimal, full rlm, dspy.RLM. [docs/sanjaya-vs-rlm-implementations.md](https://github.com/pratos/sanjaya/blob/main/docs/sanjaya-vs-rlm-implementations.md)
- Sanjaya source code. [github.com/pratos/sanjaya](https://github.com/pratos/sanjaya)
