"use client";

import { useEffect, useRef } from "react";
import { Panel } from "./panel";
import type { VisionEntry } from "@/lib/types";

interface VisionPanelProps {
  queries: VisionEntry[];
}

function formatNumber(n: number): string {
  return n.toLocaleString();
}

export function VisionPanel({ queries }: VisionPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [queries.length]);

  return (
    <Panel title="VISION QUERIES">
      {queries.length === 0 ? (
        <span className="text-[10px] uppercase tracking-wider text-hud-dim">
          NO DATA YET
        </span>
      ) : (
        <div ref={scrollRef} className="space-y-2 text-[10px] overflow-auto max-h-full">
          {queries.map((v, i) => (
            <div key={i} className="border border-hud-border p-2">
              <div className="text-hud-dim mb-1 truncate">
                <span className="text-[9px] text-hud-label uppercase tracking-wider">
                  PROMPT:{" "}
                </span>
                {v.prompt}
              </div>
              <div className="text-foreground mb-1 truncate">{v.responsePreview}</div>
              <div className="flex gap-2 text-hud-dim">
                <span className="text-hud-cyan">
                  {v.frameCount}f / {v.clipCount}c
                </span>
                {v.modelUsed && (
                  <span className="text-hud-magenta">{v.modelUsed}</span>
                )}
                {v.inputTokens != null && (
                  <span>IN: {formatNumber(v.inputTokens)}</span>
                )}
                {v.outputTokens != null && (
                  <span>OUT: {formatNumber(v.outputTokens)}</span>
                )}
                {v.durationSeconds != null && (
                  <span className="tabular-nums">{v.durationSeconds.toFixed(1)}s</span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}
