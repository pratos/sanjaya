"use client";

import { useEffect, useState } from "react";
import { CheckCircle, XCircle, Clock } from "lucide-react";
import { Panel } from "./panel";
import { RunDetailModal } from "./run-detail-modal";
import { fetchHistory, type HistoryEntry } from "@/lib/api";

function formatTimestamp(ts: number | null): string {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  return d.toLocaleString("en-GB", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function videoName(path: string | null): string {
  if (!path) return "unknown";
  const parts = path.split("/");
  return parts[parts.length - 1];
}

function StatusIcon({ status }: { status: HistoryEntry["status"] }) {
  switch (status) {
    case "complete":
      return <CheckCircle size={14} className="text-hud-green shrink-0" />;
    case "error":
      return <XCircle size={14} className="text-hud-red shrink-0" />;
    default:
      return <Clock size={14} className="text-hud-dim shrink-0" />;
  }
}

import type { RunStatus } from "@/lib/types";

export function RunHistory({ runStatus }: { runStatus: RunStatus }) {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [detailRunId, setDetailRunId] = useState<string | null>(null);

  // Fetch on mount and whenever a run completes/errors
  useEffect(() => {
    fetchHistory()
      .then(setEntries)
      .finally(() => setLoading(false));
  }, [runStatus]);

  return (
    <Panel
      title="RUN HISTORY"
      status={loading ? "active" : undefined}
      className=""
    >
      <div className="h-full overflow-y-auto">
        {loading && (
          <p className="text-sm text-hud-dim p-2">Loading history…</p>
        )}
        {!loading && entries.length === 0 && (
          <p className="text-sm text-hud-dim p-2">No runs found.</p>
        )}
        {entries.map((entry) => {
          return (
            <div
              key={entry.runId}
              className="border-b border-hud-border last:border-b-0"
            >
              {/* Summary row */}
              <button
                onClick={() => setDetailRunId(entry.runId)}
                className="w-full flex items-center gap-3 px-3 py-2.5 text-left text-sm transition-colors hover:bg-[#141414]"
              >
                <StatusIcon status={entry.status} />
                <span className="font-mono text-xs text-hud-dim w-[7ch] shrink-0">
                  {entry.runId.slice(-6)}
                </span>
                <span className="truncate flex-1 text-sm text-foreground/90">
                  {entry.question ?? "—"}
                </span>
                <span className="text-xs text-hud-dim shrink-0">
                  {videoName(entry.videoPath)}
                </span>
                <span className="text-xs text-hud-dim shrink-0 w-[10ch] text-right">
                  {formatTimestamp(entry.timestamp)}
                </span>
              </button>

              
            </div>
          );
        })}
      </div>

      {detailRunId && (
        <RunDetailModal
          runId={detailRunId}
          onClose={() => setDetailRunId(null)}
        />
      )}
    </Panel>
  );
}


