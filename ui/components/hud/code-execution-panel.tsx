"use client";

import { useEffect, useRef } from "react";
import { Panel } from "./panel";
import type { CodeExecution } from "@/lib/types";

interface CodeExecutionPanelProps {
  executions: CodeExecution[];
  isRunning: boolean;
}

export function CodeExecutionPanel({
  executions,
  isRunning,
}: CodeExecutionPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to latest entry
  useEffect(() => {
    const el = scrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [executions.length]);

  return (
    <Panel
      title="CODE EXECUTION LOG"
      status={isRunning && executions.length > 0 ? "active" : undefined}
      className="h-full flex flex-col"
    >
      {executions.length === 0 ? (
        <span className="text-[10px] uppercase tracking-wider text-hud-dim">
          NO DATA YET
        </span>
      ) : (
        <div ref={scrollRef} className="space-y-2 text-[10px] overflow-y-auto flex-1 min-h-0">
          {executions.map((exec, i) => (
            <div key={i} className="border border-hud-border">
              {/* Header */}
              <div className="flex items-center justify-between px-2 py-1 border-b border-hud-border bg-[#111]">
                <span className="text-hud-label uppercase tracking-wider">
                  CODE BLOCK {exec.codeBlockIndex}/{exec.codeBlockTotal} —
                  ITERATION {exec.iteration}
                </span>
                <span className="tabular-nums text-hud-dim">
                  {exec.executionTime.toFixed(2)}s
                </span>
              </div>
              {/* Code */}
              <pre className="px-2 py-1 text-hud-green/70 whitespace-pre-wrap break-all max-h-24 overflow-auto">
                {exec.code.length > 500
                  ? exec.code.slice(0, 500) + "…"
                  : exec.code}
              </pre>
              {/* Stderr */}
              {exec.stderr && (
                <div className="border-t border-hud-border px-2 py-1">
                  <span className="text-[9px] text-hud-red uppercase tracking-wider">
                    STDERR:{" "}
                  </span>
                  <pre className="text-hud-red/70 whitespace-pre-wrap break-all max-h-16 overflow-auto inline">
                    {exec.stderr}
                  </pre>
                </div>
              )}
              {/* Final answer indicator */}
              {exec.hasFinalAnswer && (
                <div className="border-t border-hud-border px-2 py-1 bg-hud-green/10">
                  <span className="text-hud-green text-[9px] uppercase tracking-wider font-bold">
                    ✓ FINAL ANSWER DETECTED
                  </span>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}
