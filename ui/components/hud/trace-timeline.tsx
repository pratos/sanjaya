"use client";

import { useEffect, useRef, useState } from "react";
import { Panel } from "./panel";
import { frameUrl } from "@/lib/api";
import type { TraceEvent } from "@/lib/types";

interface TraceTimelineProps {
  events: TraceEvent[];
  startTime: number | null;
}

function getEventColor(kind: string): string {
  switch (kind) {
    case "run_start":
    case "run_end":
      return "text-foreground";
    case "root_response":
    case "root_response_start":
      return "text-hud-blue";
    case "code_execution":
    case "code_instruction":
      return "text-hud-green";
    case "iteration_start":
    case "iteration_end":
      return "text-hud-amber";
    case "retrieval":
      return "text-hud-amber";
    case "clip":
    case "frames":
      return "text-hud-cyan";
    case "vision":
      return "text-hud-magenta";
    case "sub_llm":
    case "sub_llm_start":
      return "text-hud-dim";
    case "tool_call":
    case "tool_call_start":
      return "text-hud-cyan";
    case "schema_generation":
    case "schema_generation_start":
      return "text-hud-label";
    case "critic_evaluation":
      return "text-hud-amber";
    case "transcription":
      return "text-hud-label";
    default:
      return "text-hud-dim";
  }
}

function getDotColor(kind: string): string {
  switch (kind) {
    case "run_start":
    case "run_end":
      return "bg-foreground";
    case "root_response":
    case "root_response_start":
      return "bg-hud-blue";
    case "code_execution":
    case "code_instruction":
      return "bg-hud-green";
    case "iteration_start":
    case "iteration_end":
      return "bg-hud-amber";
    case "retrieval":
      return "bg-hud-amber";
    case "clip":
    case "frames":
      return "bg-hud-cyan";
    case "vision":
      return "bg-hud-magenta";
    case "sub_llm":
    case "sub_llm_start":
      return "bg-hud-dim";
    case "tool_call":
    case "tool_call_start":
      return "bg-hud-cyan";
    case "critic_evaluation":
      return "bg-hud-amber";
    default:
      return "bg-hud-dim";
  }
}

function getEventSummary(event: TraceEvent): string {
  const p = event.payload;
  if (!p) return "";
  try {
    switch (event.kind) {
      case "run_start":
        return `${(p.model as string) ?? (p.orchestrator_model as string) ?? ""} — ${p.question ? `"${(p.question as string).slice(0, 60)}"` : ""}`;
      case "run_end":
        return `status=${p.status ?? "complete"} tokens=${p.input_tokens ?? "?"}/${p.output_tokens ?? "?"}`;
      case "iteration_start":
        return `iteration ${p.iteration ?? "?"}`;
      case "iteration_end":
        return `iteration ${p.iteration ?? "?"} done${p.final_answer ? " ✓FINAL" : ""}${p.critic_rejected ? " ✗CRITIC" : ""}`;
      case "root_response_start":
        return `model=${p.model ?? "?"}`;
      case "root_response":
        return `model=${p.model ?? "?"} ${p.input_tokens ? `IN:${p.input_tokens}` : ""} ${p.output_tokens ? `OUT:${p.output_tokens}` : ""}`;
      case "code_instruction":
        return `code_preview=${((p.code_preview as string) ?? "").slice(0, 60)}`;
      case "code_execution":
        return `block=${p.block_index ?? "?"} ${(p.execution_time_s as number)?.toFixed(2) ?? (p.execution_time as number)?.toFixed(2) ?? "?"}s tools=[${(p.tools_used as string[])?.join(", ") ?? ""}]${p.final_answer ? " ✓FINAL" : ""}`;
      case "tool_call":
      case "tool_call_start":
        return `${p.tool_name ?? "?"}`;
      case "retrieval":
        return `subtitle=${p.subtitle_count ?? 0} sliding=${p.sliding_count ?? 0} selected=${p.selected_count ?? 0}`;
      case "clip":
        return `${p.clip_id ?? "?"} [${(p.start_s as number)?.toFixed(1) ?? "?"}s - ${(p.end_s as number)?.toFixed(1) ?? "?"}s]`;
      case "frames":
        return `${p.clip_id ?? "?"} → ${p.frame_count ?? 0} frames`;
      case "vision":
        return `${p.frame_count ?? 0}f/${p.clip_count ?? 0}c${p.model_used ? ` model=${p.model_used}` : (p.model as string) ? ` model=${p.model}` : ""}`;
      case "sub_llm":
        return `model=${p.model ?? "?"} ${((p.response_preview as string) ?? (p.prompt_preview as string) ?? "").slice(0, 60)}`;
      case "sub_llm_start":
        return `model=${p.model ?? "?"} ${p.prompt_chars ?? "?"}ch`;
      case "schema_generation":
      case "schema_generation_start":
        return `question=${p.question_chars ?? "?"}ch`;
      case "critic_evaluation":
        return `score=${p.score ?? "?"}/100 ${p.pass ? "✓PASS" : "✗FAIL"} ${((p.feedback as string) ?? "").slice(0, 50)}`;
      case "transcription":
        return `source=${p.source ?? "?"} path=${p.subtitle_path ?? "none"}`;
      default:
        return JSON.stringify(p ?? {}).slice(0, 80);
    }
  } catch {
    return event.kind;
  }
}

/** Keys to show as code/pre blocks instead of inline values. */
const CODE_KEYS = new Set([
  "code", "code_preview", "code_content",
  "response_content", "response_preview",
  "prompt_content", "prompt_preview",
  "stdout", "stderr",
  "final_answer",
  "feedback",
  "llm_queries_preview",
]);

/** Keys to skip in the detail view (noise or redundant). */
const SKIP_KEYS = new Set([
  "prompt_chars", "response_chars", "code_chars",
  "question_chars", "n_frames",
]);

/** Keys that contain image paths to render as thumbnails. */
const IMAGE_PATH_KEYS = new Set([
  "frame_paths",
]);

function EventDetail({ payload }: { payload: Record<string, unknown> }) {
  const entries = Object.entries(payload).filter(
    ([k, v]) => !SKIP_KEYS.has(k) && v != null && v !== ""
  );

  if (entries.length === 0) {
    return (
      <span className="text-hud-dim italic">no payload data</span>
    );
  }

  return (
    <div className="space-y-1.5">
      {entries.map(([key, value]) => {
        // Render frame_paths as image thumbnails
        if (IMAGE_PATH_KEYS.has(key) && Array.isArray(value) && value.length > 0) {
          return (
            <div key={key}>
              <span className="text-hud-dim uppercase tracking-wider">{key}: </span>
              <div className="flex gap-1 overflow-x-auto py-1 mt-0.5">
                {(value as string[]).map((path, i) => (
                  <img
                    key={i}
                    src={frameUrl(path)}
                    alt={`frame ${i + 1}`}
                    className="h-20 w-auto shrink-0 border border-hud-border/50 object-cover"
                    loading="lazy"
                  />
                ))}
              </div>
            </div>
          );
        }

        const isCode = CODE_KEYS.has(key);
        const isLongString = typeof value === "string" && value.length > 100;
        const isObject = typeof value === "object" && !Array.isArray(value);
        const isArray = Array.isArray(value);

        return (
          <div key={key}>
            <span className="text-hud-dim uppercase tracking-wider">{key}: </span>
            {isCode || isLongString ? (
              <pre className="mt-0.5 px-2 py-1 bg-[#111] border border-hud-border/50 text-foreground/80 whitespace-pre-wrap break-all max-h-48 overflow-auto">
                {String(value)}
              </pre>
            ) : isObject || isArray ? (
              <pre className="mt-0.5 px-2 py-1 bg-[#111] border border-hud-border/50 text-foreground/80 whitespace-pre-wrap break-all max-h-48 overflow-auto">
                {JSON.stringify(value, null, 2)}
              </pre>
            ) : (
              <span className="text-foreground/80">
                {typeof value === "number"
                  ? Number.isInteger(value) ? value : value.toFixed(4)
                  : String(value)}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

/** Event kinds worth showing in the timeline. Noisy _start events are hidden. */
const VISIBLE_KINDS = new Set([
  "run_start",
  "run_end",
  "iteration_start",
  "iteration_end",
  "root_response_start",
  "root_response",
  "code_instruction",
  "code_execution",
  "sub_llm_start",
  "sub_llm",
  "vision_start",
  "vision",
  "tool_call",
  "schema_generation",
  "critic_evaluation",
  "transcription",
  "retrieval",
  "clip",
  "frames",
]);

export function TraceTimeline({ events, startTime }: TraceTimelineProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  const visibleEvents = events.filter((e) => VISIBLE_KINDS.has(e.kind));

  useEffect(() => {
    const el = scrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [visibleEvents.length]);

  return (
    <Panel title="TRACE TIMELINE" className="h-full flex flex-col">
      {visibleEvents.length === 0 ? (
        <span className="text-[12px] uppercase tracking-wider text-hud-dim">
          NO DATA YET
        </span>
      ) : (
        <div
          ref={scrollRef}
          className="space-y-0 text-[12px] overflow-y-auto flex-1 min-h-0"
        >
          {visibleEvents.map((event, i) => {
            const diff = startTime && event.timestamp ? event.timestamp - startTime : 0;
            const relativeS = Number.isFinite(diff) ? diff.toFixed(1) : "0.0";
            const isExpanded = expandedIndex === i;
            const hasPayload = event.payload && Object.keys(event.payload).length > 0;

            return (
              <div
                key={i}
                className="border-b border-hud-border/30 last:border-0"
              >
                {/* Summary row */}
                <button
                  type="button"
                  onClick={() => hasPayload && setExpandedIndex(isExpanded ? null : i)}
                  className={`w-full flex items-center gap-2 py-0.5 text-left ${
                    hasPayload ? "cursor-pointer hover:bg-[#141414]" : "cursor-default"
                  }`}
                >
                  {/* Timestamp */}
                  <span className="text-hud-dim tabular-nums w-14 shrink-0 text-right leading-none">
                    +{relativeS}s
                  </span>
                  {/* Dot */}
                  <span
                    className={`h-1.5 w-1.5 shrink-0 ${getDotColor(event.kind)}`}
                  />
                  {/* Kind */}
                  <span
                    className={`uppercase shrink-0 w-32 font-bold tracking-wider leading-none ${getEventColor(event.kind)}`}
                  >
                    {event.kind}
                  </span>
                  {/* Summary */}
                  <span className="text-hud-dim truncate leading-none flex-1">
                    {getEventSummary(event)}
                  </span>
                  {/* Expand indicator */}
                  {hasPayload && (
                    <span className="text-hud-dim shrink-0 w-4 text-center">
                      {isExpanded ? "−" : "+"}
                    </span>
                  )}
                </button>

                {/* Expanded detail */}
                {isExpanded && event.payload && (
                  <div className="ml-[4.5rem] mr-2 mb-1.5 mt-0.5 p-2 border border-hud-border/50 bg-[#0d0d0d]">
                    <EventDetail payload={event.payload} />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </Panel>
  );
}
