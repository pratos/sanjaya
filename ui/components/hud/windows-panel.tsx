"use client";

import { Panel } from "./panel";
import type { TraceEvent } from "@/lib/types";

interface WindowsPanelProps {
  events: TraceEvent[];
}

interface WindowRow {
  windowId: string;
  strategy: string;
  startS: number;
  endS: number;
  score: number;
}

function deriveWindows(events: TraceEvent[]): {
  windows: WindowRow[];
  retrievalInfo: { subtitleCount: number; slidingCount: number; selectedCount: number } | null;
} {
  // Look for retrieval events to get counts
  const retrievalEvent = events.find((e) => e.kind === "retrieval");
  const retrievalInfo = retrievalEvent
    ? {
        subtitleCount: (retrievalEvent.payload.subtitle_count as number) ?? 0,
        slidingCount: (retrievalEvent.payload.sliding_count as number) ?? 0,
        selectedCount: (retrievalEvent.payload.selected_count as number) ?? 0,
      }
    : null;

  // Derive windows from clip events (window_id extracted from clip_id patterns)
  // Since retrieval events don't include individual window details in the trace,
  // we reconstruct from clip events
  const windows: WindowRow[] = [];
  const clipEvents = events.filter((e) => e.kind === "clip");

  for (const e of clipEvents) {
    const p = e.payload;
    const clipId = (p.clip_id as string) ?? "";
    const isSubtitle = clipId.startsWith("sub-");
    windows.push({
      windowId: clipId,
      strategy: isSubtitle ? "SUBTITLE" : "SLIDING",
      startS: (p.start_s as number) ?? 0,
      endS: (p.end_s as number) ?? 0,
      score: 0, // Score not available from clip events
    });
  }

  return { windows, retrievalInfo };
}

export function WindowsPanel({ events }: WindowsPanelProps) {
  const { windows, retrievalInfo } = deriveWindows(events);

  return (
    <Panel title="CANDIDATE WINDOWS">
      {retrievalInfo && (
        <div className="flex gap-3 mb-2 text-[10px] text-hud-dim">
          <span>
            SUB: <span className="text-hud-blue">{retrievalInfo.subtitleCount}</span>
          </span>
          <span>
            SLIDE: <span className="text-hud-amber">{retrievalInfo.slidingCount}</span>
          </span>
          <span>
            SEL: <span className="text-foreground">{retrievalInfo.selectedCount}</span>
          </span>
        </div>
      )}
      {windows.length === 0 ? (
        <span className="text-[10px] uppercase tracking-wider text-hud-dim">
          NO DATA YET
        </span>
      ) : (
        <div className="space-y-0">
          {/* Header */}
          <div className="grid grid-cols-[1fr_80px_60px_60px] gap-1 text-[9px] text-hud-dim uppercase tracking-wider border-b border-hud-border pb-1 mb-1">
            <span>WINDOW</span>
            <span>STRATEGY</span>
            <span className="text-right">START</span>
            <span className="text-right">END</span>
          </div>
          {/* Rows */}
          {windows.map((w) => (
            <div
              key={w.windowId}
              className="grid grid-cols-[1fr_80px_60px_60px] gap-1 text-[10px] py-0.5"
            >
              <span className="text-foreground truncate">{w.windowId}</span>
              <span
                className={
                  w.strategy === "SUBTITLE"
                    ? "text-hud-blue"
                    : "text-hud-amber"
                }
              >
                {w.strategy}
              </span>
              <span className="text-right tabular-nums text-hud-dim">
                {w.startS.toFixed(1)}s
              </span>
              <span className="text-right tabular-nums text-hud-dim">
                {w.endS.toFixed(1)}s
              </span>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}
