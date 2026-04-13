"use client";

import { useEffect, useRef } from "react";
import { Panel } from "./panel";
import { frameUrl } from "@/lib/api";
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
    <Panel title="VISION QUERIES" className="h-full flex flex-col">
      {queries.length === 0 ? (
        <span className="text-[12px] uppercase tracking-wider text-hud-dim">
          NO DATA YET
        </span>
      ) : (
        <div ref={scrollRef} className="space-y-2 text-[12px] overflow-y-auto flex-1 min-h-0">
          {queries.map((v, i) => (
            <div key={i} className="border border-hud-border p-2 space-y-1.5">
              {/* Prompt */}
              <div className="text-hud-dim truncate">
                <span className="text-[13px] text-hud-label uppercase tracking-wider">
                  PROMPT:{" "}
                </span>
                {v.prompt}
              </div>

              {/* Frame thumbnails */}
              {v.framePaths.length > 0 && (
                <div className="flex gap-1 overflow-x-auto py-1">
                  {v.framePaths.map((path, fi) => (
                    <img
                      key={fi}
                      src={frameUrl(path)}
                      alt={`frame ${fi + 1}`}
                      className="h-14 w-auto shrink-0 border border-hud-border/50 object-cover"
                      loading="lazy"
                    />
                  ))}
                </div>
              )}

              {/* Response */}
              {v.responsePreview && (
                <div className="text-foreground truncate">{v.responsePreview}</div>
              )}

              {/* Metadata */}
              <div className="flex gap-2 text-hud-dim">
                <span className="text-hud-cyan">
                  {v.frameCount}f / {v.clipCount}c
                </span>
                {v.clipId && (
                  <span className="text-hud-amber">{v.clipId}</span>
                )}
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
