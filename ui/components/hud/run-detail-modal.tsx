"use client";

import { useEffect, useState } from "react";
import { X, ExternalLink, ChevronDown, ChevronRight } from "lucide-react";
import { fetchRunManifest, frameUrl, type RunManifest } from "@/lib/api";

const LOGFIRE_BASE = "https://logfire-us.pydantic.dev/prthmesh/video-rlm";

interface RunDetailModalProps {
  runId: string;
  onClose: () => void;
}

function formatDuration(events: RunManifest["trace_events"]): string {
  if (events.length < 2) return "—";
  const first = events[0].timestamp;
  const last = events[events.length - 1].timestamp;
  const s = last - first;
  if (s < 60) return `${s.toFixed(1)}s`;
  return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
}

export function RunDetailModal({ runId, onClose }: RunDetailModalProps) {
  const [manifest, setManifest] = useState<RunManifest | null>(null);
  const [loading, setLoading] = useState(true);
  const [traceOpen, setTraceOpen] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetchRunManifest(runId)
      .then(setManifest)
      .finally(() => setLoading(false));
  }, [runId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const events = manifest?.trace_events ?? [];
  const runStart = events.find((e) => e.kind === "run_start");
  const runEnd = events.find((e) => e.kind === "run_end");
  const logfireRunId = (runStart?.payload?.run_id as string) ?? null;
  const startTs = runStart?.timestamp ?? 0;
  const status = (runEnd?.payload?.status as string) ?? "incomplete";
  const answer =
    (runEnd?.payload?.answer_full as string) ??
    (runEnd?.payload?.answer_preview as string) ??
    null;

  // Stats
  let totalTokens = 0;
  let costUsd = 0;
  let maxIter = 0;
  for (const e of events) {
    const p = e.payload;
    if (!p) continue;
    totalTokens += ((p.input_tokens as number) ?? 0) + ((p.output_tokens as number) ?? 0);
    costUsd += (p.cost_usd as number) ?? 0;
    if (e.kind === "root_response") {
      const iter = (p.iteration as number) ?? 0;
      if (iter > maxIter) maxIter = iter;
    }
  }

  // Frames grouped by clip
  const allFrames: { clipId: string; startS: number; endS: number; paths: string[] }[] = [];
  if (manifest) {
    for (const [clipId, clip] of Object.entries(manifest.clips)) {
      if (clip.frame_paths?.length) {
        allFrames.push({ clipId, startS: clip.start_s, endS: clip.end_s, paths: clip.frame_paths });
      }
    }
    allFrames.sort((a, b) => a.startS - b.startS);
  }
  const totalFrames = allFrames.reduce((n, c) => n + c.paths.length, 0);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="relative flex h-[90vh] w-[80vw] max-w-4xl flex-col border border-hud-border bg-[#0a0a0a]">
        {/* Header bar */}
        <div className="flex items-center justify-between border-b border-hud-border px-5 py-3 shrink-0">
          <div className="flex items-center gap-3">
            <span className={`text-xs font-bold uppercase tracking-wider ${
              status === "final_answer" ? "text-hud-green" : status === "incomplete" ? "text-hud-amber" : "text-hud-red"
            }`}>
              {status === "final_answer" ? "COMPLETE" : status.toUpperCase()}
            </span>
            <span className="text-hud-dim">·</span>
            <span className="font-mono text-xs text-hud-dim">{runId}</span>
            {logfireRunId && (
              <>
                <span className="text-hud-dim">·</span>
                <a
                  href={`${LOGFIRE_BASE}/search?q=run_id%3D${logfireRunId}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1 text-xs text-hud-blue hover:underline"
                >
                  <ExternalLink size={11} /> Logfire
                </a>
              </>
            )}
          </div>
          <button onClick={onClose} className="text-hud-dim hover:text-foreground"><X size={16} /></button>
        </div>

        {loading ? (
          <div className="flex-1 flex items-center justify-center">
            <p className="text-sm text-hud-dim">Loading…</p>
          </div>
        ) : !manifest ? (
          <div className="flex-1 flex items-center justify-center">
            <p className="text-sm text-hud-red">Manifest not found.</p>
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto">
            {/* Question */}
            <div className="px-5 pt-5 pb-3">
              <p className="text-xs font-bold uppercase tracking-[0.15em] text-hud-dim mb-1">QUESTION</p>
              <p className="text-base text-foreground">
                {(runStart?.payload?.question_preview as string) ?? "—"}
              </p>
            </div>

            {/* Stats row */}
            <div className="px-5 pb-4 flex gap-6">
              <Stat label="Video" value={videoName((runStart?.payload?.video_path as string) ?? "")} />
              <Stat label="Iterations" value={String(maxIter)} />
              <Stat label="Duration" value={formatDuration(events)} />
              <Stat label="Tokens" value={totalTokens.toLocaleString()} />
              {costUsd > 0 && <Stat label="Cost" value={`$${costUsd.toFixed(4)}`} />}
              <Stat label="Model" value={shortModel((runStart?.payload?.orchestrator_model as string) ?? "—")} />
            </div>

            {/* Answer */}
            {answer && (
              <div className="mx-5 mb-4 border border-hud-border bg-[#0d0d0d] p-4">
                <p className="text-xs font-bold uppercase tracking-[0.15em] text-hud-dim mb-2">ANSWER</p>
                <p className="text-sm text-foreground/90 leading-relaxed whitespace-pre-wrap">{answer}</p>
              </div>
            )}

            {/* Frames gallery */}
            {totalFrames > 0 && (
              <div className="mx-5 mb-4">
                <p className="text-xs font-bold uppercase tracking-[0.15em] text-hud-dim mb-3">
                  FRAMES <span className="text-hud-amber ml-1">{totalFrames}</span>
                </p>
                {allFrames.map((clip) => (
                  <div key={clip.clipId} className="mb-3">
                    <p className="text-[10px] uppercase tracking-wider text-hud-dim mb-1.5">
                      {clip.clipId}
                      <span className="ml-2 text-hud-amber">
                        {clip.startS.toFixed(0)}s – {clip.endS.toFixed(0)}s
                      </span>
                    </p>
                    <div className="grid grid-cols-4 gap-1.5">
                      {clip.paths.map((fp, j) => (
                        <div key={j} className="border border-hud-border/50 bg-black overflow-hidden">
                          <img
                            src={frameUrl(fp)}
                            alt={`${clip.clipId} frame ${j + 1}`}
                            className="w-full h-auto block"
                            loading="lazy"
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Trace events — collapsible */}
            <div className="mx-5 mb-5 border border-hud-border">
              <button
                onClick={() => setTraceOpen(!traceOpen)}
                className="w-full flex items-center gap-2 px-4 py-2.5 text-left hover:bg-[#111] transition-colors"
              >
                {traceOpen ? <ChevronDown size={14} className="text-hud-dim" /> : <ChevronRight size={14} className="text-hud-dim" />}
                <span className="text-xs font-bold uppercase tracking-[0.15em] text-hud-dim">
                  TRACE EVENTS
                </span>
                <span className="text-xs text-hud-amber">{events.length}</span>
              </button>
              {traceOpen && (
                <div className="border-t border-hud-border max-h-80 overflow-y-auto">
                  {events.map((event, i) => {
                    const relS = startTs ? (event.timestamp - startTs).toFixed(1) : "0.0";
                    return <TraceRow key={i} event={event} relS={relS} />;
                  })}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-[10px] uppercase tracking-[0.1em] text-hud-dim">{label}</p>
      <p className="text-xs font-mono text-foreground/80 mt-0.5">{value}</p>
    </div>
  );
}

function videoName(path: string): string {
  if (!path) return "—";
  const parts = path.split("/");
  return parts[parts.length - 1];
}

function shortModel(model: string): string {
  // "openrouter:openai/gpt-5.3-codex" → "gpt-5.3-codex"
  const parts = model.split("/");
  return parts[parts.length - 1];
}

const KIND_COLORS: Record<string, string> = {
  run_start: "text-foreground",
  run_end: "text-foreground",
  root_response: "text-hud-blue",
  code_execution: "text-hud-green",
  code_instruction: "text-hud-green",
  retrieval: "text-hud-amber",
  clip: "text-cyan-400",
  frames: "text-cyan-400",
  vision: "text-purple-400",
  sub_llm: "text-hud-dim",
  transcription: "text-hud-dim",
};

function TraceRow({ event, relS }: { event: RunManifest["trace_events"][0]; relS: string }) {
  const [open, setOpen] = useState(false);
  const p = event.payload ?? {};
  const color = KIND_COLORS[event.kind] ?? "text-hud-dim";

  let summary = "";
  try {
    switch (event.kind) {
      case "run_start": summary = `model=${p.orchestrator_model ?? "?"}`; break;
      case "run_end": summary = `status=${p.status ?? "?"}`; break;
      case "root_response": summary = `iter=${p.iteration ?? "?"} blocks=${p.code_blocks_count ?? 0}`; break;
      case "code_execution": summary = `iter=${p.iteration ?? "?"} ${(p.execution_time as number)?.toFixed(1) ?? "?"}s${p.has_final_answer ? " ✓FINAL" : ""}`; break;
      case "code_instruction": summary = `iter=${p.iteration ?? "?"} ${(p.code_chars as number) ?? 0}ch`; break;
      case "retrieval": summary = `selected=${p.selected_count ?? 0}`; break;
      case "clip": summary = `${p.clip_id ?? "?"} [${(p.start_s as number)?.toFixed(0) ?? "?"}–${(p.end_s as number)?.toFixed(0) ?? "?"}s]`; break;
      case "frames": summary = `${p.clip_id ?? "?"} → ${p.frame_count ?? 0}f`; break;
      case "vision": summary = `${p.frame_count ?? 0}f ${(p.duration_seconds as number)?.toFixed(1) ?? "?"}s`; break;
      case "sub_llm": summary = ((p.prompt_preview as string) ?? "").slice(0, 50); break;
      case "transcription": summary = `source=${p.source ?? "?"}${p.error ? ` err` : ""}`; break;
      default: summary = ""; break;
    }
  } catch { summary = ""; }

  const expandable = ["code_execution", "code_instruction", "vision", "sub_llm", "run_end"].includes(event.kind);

  return (
    <div className="border-b border-hud-border/20 last:border-0">
      <button
        onClick={() => expandable && setOpen(!open)}
        className={`w-full flex items-center gap-2 px-4 py-1.5 text-[11px] text-left ${expandable ? "hover:bg-[#111] cursor-pointer" : "cursor-default"}`}
      >
        <span className="font-mono text-hud-dim w-[5ch] text-right shrink-0">+{relS}s</span>
        <span className={`uppercase font-bold w-[10ch] shrink-0 ${color}`}>{event.kind.replace("_", " ")}</span>
        <span className="text-foreground/50 truncate">{summary}</span>
      </button>
      {open && expandable && (
        <div className="px-4 pb-2 pt-0.5 ml-[7ch] text-xs">
          {(event.kind === "code_execution" || event.kind === "code_instruction") && (
            <pre className="text-hud-green/70 bg-[#111] border border-hud-border/50 p-2 overflow-x-auto whitespace-pre-wrap max-h-48 overflow-y-auto">
              {(p.code_content as string) ?? (p.code_preview as string) ?? "—"}
            </pre>
          )}
          {event.kind === "vision" && (
            <>
              <p className="text-foreground/50 mb-1">{(p.prompt as string) ?? "—"}</p>
              <p className="text-foreground/80 whitespace-pre-wrap">{(p.response_preview as string) ?? "—"}</p>
            </>
          )}
          {event.kind === "sub_llm" && (
            <>
              <p className="text-foreground/50 mb-1">{(p.prompt_preview as string) ?? "—"}</p>
              <p className="text-foreground/80 whitespace-pre-wrap">{(p.response_preview as string) ?? "—"}</p>
            </>
          )}
          {event.kind === "run_end" && (
            <p className="text-foreground/80 whitespace-pre-wrap">
              {(p.answer_full as string) ?? (p.answer_preview as string) ?? "No answer."}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
