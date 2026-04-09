"use client";

import { useEffect, useState } from "react";
import type { BenchmarkData } from "@/lib/types";
import { fetchBenchmarks } from "@/lib/api";
import { SummaryHeader } from "@/components/benchmark/summary-header";
import { OverviewTable } from "@/components/benchmark/overview-table";
import { LiveRunHistory } from "@/components/benchmark/live-run-history";

export default function BenchmarkDashboard() {
  const [data, setData] = useState<BenchmarkData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchBenchmarks()
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  if (error) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="border border-hud-red bg-hud-red/5 px-6 py-4">
          <span className="text-sm text-hud-red">{error}</span>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <span className="text-xs text-hud-dim uppercase tracking-[0.2em] animate-pulse">
          Loading benchmark data...
        </span>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen flex-col">
      <SummaryHeader data={data} />
      <div className="flex-1 p-4 space-y-4">
        <OverviewTable prompts={data.prompts} latestVersion={data.summary.latestVersion} />
        <LiveRunHistory data={data.liveRuns} videos={data.videos} />
      </div>
    </div>
  );
}
