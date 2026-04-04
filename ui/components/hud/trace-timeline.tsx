"use client";

import { useEffect, useRef } from "react";
import { Panel } from "./panel";
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
      return "text-hud-blue";
    case "code_execution":
    case "code_instruction":
      return "text-hud-green";
    case "retrieval":
      return "text-hud-amber";
    case "clip":
    case "frames":
      return "text-hud-cyan";
    case "vision":
      return "text-hud-magenta";
    case "sub_llm":
      return "text-hud-dim";
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
      return "bg-hud-blue";
    case "code_execution":
    case "code_instruction":
      return "bg-hud-green";
    case "retrieval":
      return "bg-hud-amber";
    case "clip":
    case "frames":
      return "bg-hud-cyan";
    case "vision":
      return "bg-hud-magenta";
    case "sub_llm":
      return "bg-hud-dim";
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
        return `${(p.orchestrator_model as string) ?? ""} — max ${p.max_iterations ?? "?"} iters`;
      case "run_end":
        return `status=${p.status ?? "?"}`;
      case "root_response":
        return `iter=${p.iteration ?? "?"} code_blocks=${p.code_blocks_count ?? 0}${p.model_used ? ` model=${p.model_used}` : ""}`;
      case "code_execution":
        return `iter=${p.iteration ?? "?"} block=${p.code_block_index ?? "?"}/${p.code_block_total ?? "?"} ${(p.execution_time as number)?.toFixed(2) ?? "?"}s${p.has_final_answer ? " ✓FINAL" : ""}`;
      case "code_instruction":
        return `iter=${p.iteration ?? "?"} block=${p.code_block_index ?? "?"}/${p.code_block_total ?? "?"} ${(p.code_chars as number) ?? 0}ch`;
      case "retrieval":
        return `subtitle=${p.subtitle_count ?? 0} sliding=${p.sliding_count ?? 0} selected=${p.selected_count ?? 0}`;
      case "clip":
        return `${p.clip_id ?? "?"} [${(p.start_s as number)?.toFixed(1) ?? "?"}s - ${(p.end_s as number)?.toFixed(1) ?? "?"}s]`;
      case "frames":
        return `${p.clip_id ?? "?"} → ${p.frame_count ?? 0} frames`;
      case "vision":
        return `${p.frame_count ?? 0}f/${p.clip_count ?? 0}c${p.model_used ? ` model=${p.model_used}` : ""}`;
      case "sub_llm":
        return `${((p.prompt_preview as string) ?? "").slice(0, 60)}`;
      case "transcription":
        return `source=${p.source ?? "?"} path=${p.subtitle_path ?? "none"}`;
      default:
        return JSON.stringify(p ?? {}).slice(0, 80);
    }
  } catch {
    return event.kind;
  }
}

export function TraceTimeline({ events, startTime }: TraceTimelineProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [events.length]);

  return (
    <Panel title="TRACE TIMELINE" className="">
      {events.length === 0 ? (
        <span className="text-[10px] uppercase tracking-wider text-hud-dim">
          NO DATA YET
        </span>
      ) : (
        <div
          ref={scrollRef}
          className="space-y-0 text-[10px] overflow-auto max-h-full"
        >
          {events.map((event, i) => {
            const diff = startTime && event.timestamp ? event.timestamp - startTime : 0;
            const relativeS = Number.isFinite(diff) ? diff.toFixed(1) : "0.0";
            return (
              <div
                key={i}
                className="flex items-start gap-2 py-0.5 border-b border-hud-border/30 last:border-0"
              >
                {/* Timestamp */}
                <span className="text-hud-dim tabular-nums w-14 shrink-0 text-right">
                  +{relativeS}s
                </span>
                {/* Dot */}
                <span
                  className={`mt-1.5 h-1.5 w-1.5 shrink-0 ${getDotColor(event.kind)}`}
                />
                {/* Kind */}
                <span
                  className={`uppercase shrink-0 w-24 font-bold tracking-wider ${getEventColor(event.kind)}`}
                >
                  {event.kind}
                </span>
                {/* Summary */}
                <span className="text-hud-dim truncate">
                  {getEventSummary(event)}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </Panel>
  );
}
