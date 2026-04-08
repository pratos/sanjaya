import { readdir, readFile } from "fs/promises";
import { join } from "path";
import { NextResponse, type NextRequest } from "next/server";

const PROJECT_ROOT = join(process.cwd(), "..");
const ARTIFACT_DIRS = [
  join(PROJECT_ROOT, "sanjaya_artifacts"),
  join(PROJECT_ROOT, "nbs", "sanjaya_artifacts"),
];

/** Map internal tracer span names to frontend event names. */
const KIND_MAP: Record<string, string> = {
  "sanjaya.completion_start": "run_start",
  "sanjaya.completion_end": "run_end",
  "sanjaya.iteration_start": "iteration_start",
  "sanjaya.iteration_end": "iteration_end",
  "sanjaya.root_llm_call_start": "root_response_start",
  "sanjaya.root_llm_call_end": "root_response",
  "sanjaya.code_execution_start": "code_instruction",
  "sanjaya.code_execution_end": "code_execution",
  "sanjaya.tool_call_start": "tool_call_start",
  "sanjaya.tool_call_end": "tool_call",
  "sanjaya.sub_llm_call.regular_start": "sub_llm_start",
  "sanjaya.sub_llm_call.regular_end": "sub_llm",
  "sanjaya.sub_llm_call.vision_start": "vision_start",
  "sanjaya.sub_llm_call.vision_end": "vision",
  "sanjaya.sub_llm_call.caption_frames_start": "vision_start",
  "sanjaya.sub_llm_call.caption_frames_end": "vision",
  "sanjaya.schema_generation_start": "schema_generation_start",
  "sanjaya.schema_generation_end": "schema_generation",
  "sanjaya.critic_evaluation": "critic_evaluation",
};

interface RawEvent {
  kind: string;
  timestamp: number;
  [key: string]: unknown;
}

function normalizeEvent(e: RawEvent) {
  const { kind: rawKind, timestamp, ...payload } = e;
  return {
    kind: KIND_MAP[rawKind] ?? rawKind,
    timestamp,
    payload,
  };
}

/**
 * GET /api/traces?question=...
 * Returns trace events from sanjaya_artifacts/ trace.json files,
 * matched by question text. Picks the trace with the most events.
 */
export async function GET(req: NextRequest) {
  const question = req.nextUrl.searchParams.get("question");
  if (!question) {
    return NextResponse.json({ error: "Missing question param" }, { status: 400 });
  }

  const questionPrefix = question.slice(0, 80);
  let bestTrace: { events: RawEvent[]; runId: string; question: string } | null = null;

  for (const artDir of ARTIFACT_DIRS) {
    let runs: string[];
    try {
      runs = await readdir(artDir);
    } catch {
      continue;
    }

    for (const run of runs) {
      const tracePath = join(artDir, run, "trace.json");
      try {
        const raw = JSON.parse(await readFile(tracePath, "utf-8"));
        const traceQuestion = (raw.question as string) ?? "";
        if (traceQuestion.slice(0, 80) === questionPrefix) {
          const events = (raw.events as RawEvent[]) ?? [];
          if (!bestTrace || events.length > bestTrace.events.length) {
            bestTrace = { events, runId: raw.run_id ?? run, question: traceQuestion };
          }
        }
      } catch {
        // skip broken traces
      }
    }
  }

  if (!bestTrace) {
    return NextResponse.json({ events: [], runId: null });
  }

  return NextResponse.json({
    events: bestTrace.events.map(normalizeEvent),
    runId: bestTrace.runId,
  });
}
