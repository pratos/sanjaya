/** API client for the Sanjaya backend. */

import type { TraceEvent } from "./types";

export interface HistoryEntry {
  runId: string;
  timestamp: number | null;
  videoPath: string | null;
  question: string | null;
  status: "complete" | "error" | "incomplete";
  answerPreview: string | null;
  eventCount: number;
  iterations: number;
  totalTokens: number;
  costUsd: number;
  models: { orchestrator: string | null; recursive: string | null };
}

export async function fetchHistory(): Promise<HistoryEntry[]> {
  const res = await fetch("/api/history");
  if (!res.ok) return [];
  return res.json();
}

export interface RunManifest {
  run_id: string;
  clips: Record<string, {
    clip_id: string;
    clip_path: string;
    start_s: number;
    end_s: number;
    frame_paths: string[];
    window_id: string;
  }>;
  candidate_windows: Array<{
    window_id: string;
    strategy: string;
    start_s: number;
    end_s: number;
    score: number;
    reason: string;
  }>;
  trace_events: Array<{
    kind: string;
    timestamp: number;
    payload: Record<string, unknown>;
  }>;
}

export async function fetchRunManifest(runId: string): Promise<RunManifest | null> {
  const res = await fetch(`/api/history/${encodeURIComponent(runId)}`);
  if (!res.ok) return null;
  return res.json();
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface RunRequest {
  video_path: string;
  question: string;
  subtitle_path?: string;
  subtitle_mode?: string;
  subtitle_api_model?: string;
  max_iterations?: number;
}

export interface VideoEntry {
  path: string;
  hasTranscript: boolean;
}

export async function fetchVideos(): Promise<VideoEntry[]> {
  const res = await fetch("/api/videos");
  if (!res.ok) {
    throw new Error(`Failed to fetch videos: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export interface TranscriptSegment {
  start: number;
  end: number;
  text: string;
}

export async function fetchTranscript(
  videoRelPath: string
): Promise<TranscriptSegment[]> {
  const res = await fetch(`/api/videos/transcript?path=${encodeURIComponent(videoRelPath)}`);
  if (!res.ok) return [];
  const data = await res.json();
  return data.segments ?? [];
}

export function videoStreamUrl(videoRelPath: string): string {
  return `/api/videos/stream?path=${encodeURIComponent(videoRelPath)}`;
}

export function frameUrl(framePath: string): string {
  return `/api/frames?path=${encodeURIComponent(framePath)}`;
}

export async function submitRun(
  request: RunRequest
): Promise<{ run_id: string }> {
  const res = await fetch(`${API_BASE}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!res.ok) {
    throw new Error(`Failed to start run: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export function streamEvents(
  runId: string,
  onEvent: (event: TraceEvent) => void,
  onError: (error: string) => void,
  onEnd: () => void
): () => void {
  const url = `${API_BASE}/runs/${runId}/events`;
  const eventSource = new EventSource(url);

  const handleMessage = (e: MessageEvent) => {
    try {
      const parsed = JSON.parse(e.data) as TraceEvent;
      if (parsed.kind === "stream_end") {
        onEnd();
        eventSource.close();
        return;
      }
      if (parsed.kind === "stream_error") {
        onError((parsed.payload as { error?: string }).error ?? "Unknown error");
        eventSource.close();
        return;
      }
      onEvent(parsed);
    } catch {
      // ignore parse errors on heartbeats
    }
  };

  // Listen to all named event types we care about
  const eventTypes = [
    "run_start",
    "run_end",
    "transcription",
    "root_response",
    "code_instruction",
    "code_execution",
    "retrieval",
    "clip",
    "frames",
    "vision",
    "sub_llm",
    "heartbeat",
    "stream_end",
    "stream_error",
  ];

  for (const type of eventTypes) {
    eventSource.addEventListener(type, handleMessage);
  }

  // Also listen to generic messages as fallback
  eventSource.onmessage = handleMessage;

  eventSource.onerror = () => {
    onError("SSE connection error");
    eventSource.close();
  };

  // Return cleanup function
  return () => {
    eventSource.close();
  };
}
