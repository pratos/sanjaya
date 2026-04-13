"use client";

import { Panel } from "./panel";
import type { ClipEntry } from "@/lib/types";

interface ClipPanelProps {
  clips: ClipEntry[];
}

export function ClipPanel({ clips }: ClipPanelProps) {
  return (
    <Panel title="CLIPS & FRAMES">
      {clips.length === 0 ? (
        <span className="text-[12px] uppercase tracking-wider text-hud-dim">
          NO DATA YET
        </span>
      ) : (
        <div className="space-y-0">
          {/* Header */}
          <div className="grid grid-cols-[1fr_60px_60px_50px] gap-1 text-[13px] text-hud-dim uppercase tracking-wider border-b border-hud-border pb-1 mb-1">
            <span>CLIP ID</span>
            <span className="text-right">START</span>
            <span className="text-right">END</span>
            <span className="text-right">FRAMES</span>
          </div>
          {/* Rows */}
          {clips.map((clip) => (
            <div
              key={clip.clipId}
              className="grid grid-cols-[1fr_60px_60px_50px] gap-1 text-[12px] py-0.5"
            >
              <span className="text-foreground truncate">{clip.clipId}</span>
              <span className="text-right tabular-nums text-hud-dim">
                {clip.startS.toFixed(1)}s
              </span>
              <span className="text-right tabular-nums text-hud-dim">
                {clip.endS.toFixed(1)}s
              </span>
              <span className="text-right tabular-nums text-hud-cyan">
                {clip.frameCount}
              </span>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}
