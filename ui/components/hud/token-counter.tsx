"use client";

import { Panel } from "./panel";
import { DataRow } from "./data-row";
import type { TokenTotals, TraceEvent } from "@/lib/types";

interface TokenCounterProps {
  totals: TokenTotals;
  events: TraceEvent[];
  isRunning: boolean;
}

function formatNumber(n: number): string {
  return n.toLocaleString();
}

function formatCost(n: number): string {
  return `$${n.toFixed(4)}`;
}

/** Break down tokens by source category. */
function deriveBreakdown(events: TraceEvent[]) {
  let orchestratorTokens = 0;
  let subLLMTokens = 0;
  let visionTokens = 0;

  for (const e of events) {
    const p = e.payload;
    if (!p) continue;
    const total = ((p.input_tokens as number) ?? 0) + ((p.output_tokens as number) ?? 0);
    if (e.kind === "root_response") orchestratorTokens += total;
    if (e.kind === "sub_llm") subLLMTokens += total;
    if (e.kind === "vision") visionTokens += total;
  }

  return { orchestratorTokens, subLLMTokens, visionTokens };
}

export function TokenCounter({ totals, events, isRunning }: TokenCounterProps) {
  const breakdown = deriveBreakdown(events);
  const hasData = totals.totalTokens > 0;

  return (
    <Panel title="TOKEN COUNTERS" status={isRunning ? "active" : undefined}>
      <div className="space-y-2">
        <DataRow
          label="TOTAL TOKENS"
          value={hasData ? formatNumber(totals.totalTokens) : "—"}
        />
        <DataRow
          label="INPUT TOKENS"
          value={hasData ? formatNumber(totals.inputTokens) : "—"}
        />
        <DataRow
          label="OUTPUT TOKENS"
          value={hasData ? formatNumber(totals.outputTokens) : "—"}
        />
        <div className="my-2 h-px bg-hud-border" />
        <DataRow
          label="EST. COST"
          value={totals.costUsd > 0 ? formatCost(totals.costUsd) : "—"}
          valueClassName={totals.costUsd > 0 ? "text-hud-amber" : undefined}
        />
        {hasData && (
          <>
            <div className="my-2 h-px bg-hud-border" />
            <div className="text-[9px] uppercase tracking-wider text-hud-dim mb-1">
              BY SOURCE
            </div>
            <DataRow
              label="ORCHESTRATOR"
              value={formatNumber(breakdown.orchestratorTokens)}
            />
            <DataRow
              label="SUB-LLM"
              value={formatNumber(breakdown.subLLMTokens)}
            />
            <DataRow
              label="VISION"
              value={formatNumber(breakdown.visionTokens)}
            />
          </>
        )}
      </div>
    </Panel>
  );
}
