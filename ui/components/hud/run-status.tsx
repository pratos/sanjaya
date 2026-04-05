"use client";

import { useEffect, useState } from "react";
import { StatusBar } from "./status-bar";
import type { RunStatus } from "@/lib/types";

interface RunStatusHeaderProps {
  status: RunStatus;
  runId: string | null;
  iteration: number;
  maxIterations: number;
  startTime: number | null;
}

function useElapsedMs(status: RunStatus, startTime: number | null): number {
  const [elapsedMs, setElapsedMs] = useState(0);

  useEffect(() => {
    if (status !== "running" || !startTime) return;

    const interval = setInterval(() => {
      setElapsedMs(Date.now() - startTime * 1000);
    }, 100);

    return () => clearInterval(interval);
  }, [status, startTime]);

  if (status === "idle") return 0;
  return elapsedMs;
}

export function RunStatusHeader({
  status,
  runId,
  iteration,
  maxIterations,
  startTime,
}: RunStatusHeaderProps) {
  const elapsedMs = useElapsedMs(status, startTime);

  return (
    <StatusBar
      status={status}
      runId={runId}
      iteration={iteration}
      maxIterations={maxIterations}
      elapsedMs={elapsedMs}
    />
  );
}
