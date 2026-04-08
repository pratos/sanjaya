"use client";

import { useEffect, useRef } from "react";
import { Panel } from "./panel";
import type { SubLLMCall } from "@/lib/types";

interface SubLLMPanelProps {
  calls: SubLLMCall[];
}

function formatNumber(n: number): string {
  return n.toLocaleString();
}

export function SubLLMPanel({ calls }: SubLLMPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [calls.length]);

  return (
    <Panel title="SUB-LLM QUERIES" className="h-full flex flex-col">
      {calls.length === 0 ? (
        <span className="text-[10px] uppercase tracking-wider text-hud-dim">
          NO DATA YET
        </span>
      ) : (
        <div ref={scrollRef} className="space-y-2 text-[10px] overflow-y-auto flex-1 min-h-0">
          {calls.map((call, i) => (
            <div key={i} className="border border-hud-border p-2">
              {/* Prompt */}
              <div className="text-hud-dim mb-1">
                <span className="text-[9px] text-hud-label uppercase tracking-wider">
                  PROMPT:{" "}
                </span>
                <span className="truncate">{call.promptPreview}</span>
              </div>
              {/* Response */}
              <div className="text-foreground mb-1">
                <span className="text-[9px] text-hud-label uppercase tracking-wider">
                  RESPONSE:{" "}
                </span>
                <span className="truncate">{call.responsePreview}</span>
              </div>
              {/* Metadata row */}
              <div className="flex gap-3 text-hud-dim">
                {call.modelUsed && (
                  <span className="text-hud-blue">{call.modelUsed}</span>
                )}
                {call.inputTokens != null && (
                  <span>IN: {formatNumber(call.inputTokens)}</span>
                )}
                {call.outputTokens != null && (
                  <span>OUT: {formatNumber(call.outputTokens)}</span>
                )}
                {call.durationSeconds != null && (
                  <span className="tabular-nums">
                    {call.durationSeconds.toFixed(1)}s
                  </span>
                )}
                {call.costUsd != null && call.costUsd > 0 && (
                  <span className="text-hud-amber">
                    ${call.costUsd.toFixed(4)}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}
