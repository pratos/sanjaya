# Sanjaya HUD — Frontend

Live monitoring dashboard for VideoRLM orchestration runs. Dark, terminal-inspired HUD aesthetic with Berkeley Mono font throughout.

## Setup

```bash
bun install
bun dev --port 5100   # → http://localhost:5100
```

Or from project root: `just dev` (starts both UI and API via Overmind).

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | FastAPI backend URL |

## Design System

- **Background**: Near-black (`#0a0a0a` / `#0d0d0d`)
- **Borders**: Thin 1px (`#333` default, `#666` hover)
- **Text**: White (`#e5e5e5`) body, dim (`#888`) secondary
- **Accents**: Red (`#ff4444`) errors, Green (`#00ff88`) success, Amber (`#ffaa00`) active, Blue (`#4488ff`) info
- **Font**: Berkeley Mono throughout
- **Components**: shadcn/ui restyled with 0px border-radius, outlined buttons, monospace tables

## Panel Layout

```
┌──────────────────────────────────────────────────────────┐
│  STATUS BAR: RUN ID / STATUS / ITERATION / ELAPSED TIME  │
├────────────────────┬──────────────────┬──────────────────┤
│  QUERY INPUT       │  TOKEN COUNTERS  │  PROGRESS        │
├────────────────────┼──────────────────┴──────────────────┤
│  CANDIDATE WINDOWS │  CODE EXECUTION LOG                 │
├────────────────────┼─────────────────────────────────────┤
│  CLIPS & FRAMES    │  SUB-LLM QUERIES                   │
├────────────────────┼─────────────────────────────────────┤
│  VISION QUERIES    │  TRACE TIMELINE                     │
└────────────────────┴─────────────────────────────────────┘
```

## Tech Stack

- Next.js 16 (App Router, Turbopack)
- React 19
- Tailwind CSS v4
- shadcn/ui (restyled)
- SSE via native EventSource
