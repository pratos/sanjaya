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
      <footer className="border-t border-hud-border bg-hud-panel px-6 py-6">
        <span className="block text-[12px] font-bold uppercase tracking-[0.2em] text-hud-dim mb-3">
          Video Sources
        </span>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {data.videos.map((v) => (
            <a
              key={v.key}
              href={v.youtubeUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="group border border-hud-border px-3 py-2 hover:border-foreground/30 transition-colors"
            >
              <span className="block text-xs font-bold text-foreground group-hover:text-white transition-colors">
                {v.title}
              </span>
              <span className="block text-[12px] text-hud-dim mt-0.5">
                {v.channel} &middot; {v.duration}
              </span>
            </a>
          ))}
        </div>
        <p className="text-[12px] text-hud-dim/60 mt-3">
          Videos used under fair use for non-commercial research evaluation. All rights belong to the original creators.
        </p>
      </footer>
    </div>
  );
}
