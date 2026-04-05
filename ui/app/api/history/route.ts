import { readdir, readFile } from "fs/promises";
import { join, basename } from "path";
import { NextResponse } from "next/server";

const DATA_DIR = join(process.cwd(), "..", "data");

export interface HistoryEntry {
  runId: string;
  timestamp: number | null;
  videoPath: string | null;
  question: string | null;
  status: "complete" | "error" | "incomplete";
  answerPreview: string | null;
  eventCount: number;
  iterations: number;
  totalTokens: number;
  costUsd: number;
  models: { orchestrator: string | null; recursive: string | null };
}

async function scanArtifacts(dir: string): Promise<HistoryEntry[]> {
  const results: HistoryEntry[] = [];
  let topDirs;
  try {
    topDirs = await readdir(dir, { withFileTypes: true });
  } catch {
    return results;
  }

  for (const d of topDirs) {
    if (!d.isDirectory()) continue;
    const artDir = join(dir, d.name, "artifacts");
    let runDirs;
    try {
      runDirs = await readdir(artDir, { withFileTypes: true });
    } catch {
      continue;
    }

    for (const rd of runDirs) {
      if (!rd.isDirectory()) continue;
      const manifestPath = join(artDir, rd.name, "manifest.json");
      try {
        const raw = await readFile(manifestPath, "utf-8");
        const manifest = JSON.parse(raw);
        const events: Array<{ kind: string; timestamp: number; payload: Record<string, unknown> }> =
          manifest.trace_events ?? [];

        const runStart = events.find((e) => e.kind === "run_start");
        const runEnd = events.find((e) => e.kind === "run_end");

        // Tally tokens
        let totalTokens = 0;
        let costUsd = 0;
        let maxIteration = 0;
        for (const e of events) {
          const p = e.payload;
          if (p) {
            totalTokens +=
              ((p.input_tokens as number) ?? 0) +
              ((p.output_tokens as number) ?? 0);
            costUsd += (p.cost_usd as number) ?? 0;
          }
          if (e.kind === "root_response") {
            const iter = (e.payload?.iteration as number) ?? 0;
            if (iter > maxIteration) maxIteration = iter;
          }
        }

        let status: HistoryEntry["status"] = "incomplete";
        if (runEnd) {
          const s = runEnd.payload?.status as string;
          status = s === "error" ? "error" : "complete";
        }

        results.push({
          runId: manifest.run_id ?? rd.name,
          timestamp: runStart?.timestamp ?? null,
          videoPath: (runStart?.payload?.video_path as string) ?? null,
          question: (runStart?.payload?.question_preview as string) ?? null,
          status,
          answerPreview: (runEnd?.payload?.answer_preview as string) ?? null,
          eventCount: events.length,
          iterations: maxIteration,
          totalTokens,
          costUsd,
          models: {
            orchestrator: (runStart?.payload?.orchestrator_model as string) ?? null,
            recursive: (runStart?.payload?.recursive_model as string) ?? null,
          },
        });
      } catch {
        // skip broken manifests
      }
    }
  }

  return results;
}

export async function GET() {
  const entries = await scanArtifacts(DATA_DIR);
  // Sort newest first
  entries.sort((a, b) => (b.timestamp ?? 0) - (a.timestamp ?? 0));
  return NextResponse.json(entries);
}
