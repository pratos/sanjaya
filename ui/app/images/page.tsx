"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { Play } from "lucide-react";
import type { ImageBenchmarkData, ImagePrompt, ImageResult, TraceEvent } from "@/lib/types";
import { fetchImageBenchmarks, frameUrl } from "@/lib/api";
import { TraceTimeline } from "@/components/hud/trace-timeline";

function formatTime(seconds: number): string {
  if (seconds >= 60) return `${(seconds / 60).toFixed(1)}m`;
  return `${seconds.toFixed(0)}s`;
}

function formatRatio(value: number | null): string {
  if (value === null || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function collectStrings(value: unknown): string[] {
  if (typeof value === "string") return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string");
}

function fileNameFromPath(path: string): string {
  const trimmed = path.startsWith("file://") ? path.slice("file://".length) : path;
  return trimmed.split("/").filter(Boolean).pop() ?? trimmed;
}

function toEvidenceLines(value: unknown): string[] {
  if (typeof value === "string") return [value];
  if (Array.isArray(value)) return value.flatMap((item) => toEvidenceLines(item));
  if (value && typeof value === "object") {
    const item = value as Record<string, unknown>;
    const filename =
      typeof item["filename"] === "string"
        ? item["filename"]
        : typeof item["image_id"] === "string"
          ? String(item["image_id"])
          : null;
    const observations = collectStrings(item["observations"]);
    if (filename && observations.length > 0) {
      return observations.map((obs) => `${filename}: ${obs}`);
    }
    try {
      return [JSON.stringify(value)];
    } catch {
      return [];
    }
  }
  return [];
}

interface PhotobenchPromptView {
  prompt: ImagePrompt;
  result: ImageResult;
  queryId: string;
  album: string;
  predictions: string[];
  groundTruth: string[];
  predictionImagePaths: string[];
  groundTruthImagePaths: string[];
  evidenceLines: string[];
  precision: number | null;
  recall: number | null;
}

function parsePhotobenchPrompt(prompt: ImagePrompt): PhotobenchPromptView | null {
  const result = prompt.versions[prompt.bestVersion];
  if (!result) return null;

  const answerData = result.answerData && typeof result.answerData === "object"
    ? (result.answerData as Record<string, unknown>)
    : null;

  const benchmark = typeof answerData?.["benchmark"] === "string" ? answerData["benchmark"] : null;
  if (benchmark !== "PhotoBench") return null;

  const queryIdRaw =
    typeof answerData?.["query_id"] === "string"
      ? answerData["query_id"]
      : prompt.promptName.replace(/^photo_/, "");

  const albumRaw =
    typeof answerData?.["album"] === "string"
      ? answerData["album"]
      : queryIdRaw.includes("_")
        ? queryIdRaw.split("_")[0]
        : "unknown";

  const predictions =
    collectStrings(answerData?.["predictions"]).length > 0
      ? collectStrings(answerData?.["predictions"])
      : collectStrings(answerData?.["prediction"]);

  const groundTruth =
    collectStrings(answerData?.["ground_truth"]).length > 0
      ? collectStrings(answerData?.["ground_truth"])
      : collectStrings(answerData?.["groundTruth"]);

  const predictionImagePaths = collectStrings(answerData?.["prediction_image_paths"]);
  const groundTruthImagePaths = collectStrings(answerData?.["ground_truth_image_paths"]);

  const metrics = answerData?.["metrics"] && typeof answerData["metrics"] === "object"
    ? (answerData["metrics"] as Record<string, unknown>)
    : null;

  const precision = typeof metrics?.["precision"] === "number" ? metrics["precision"] : null;
  const recall = typeof metrics?.["recall"] === "number" ? metrics["recall"] : null;

  return {
    prompt,
    result,
    queryId: String(queryIdRaw),
    album: String(albumRaw),
    predictions,
    groundTruth,
    predictionImagePaths,
    groundTruthImagePaths,
    evidenceLines: toEvidenceLines(answerData?.["evidence"]),
    precision,
    recall,
  };
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col items-end gap-0.5">
      <span className="text-[12px] font-bold uppercase tracking-[0.15em] text-hud-dim">{label}</span>
      <span className="text-[12px] font-bold tabular-nums text-foreground">{value}</span>
    </div>
  );
}

function ImageGrid({
  title,
  paths,
  color,
  onOpen,
}: {
  title: string;
  paths: string[];
  color: "cyan" | "green";
  onOpen: (src: string, caption: string) => void;
}) {
  const border = color === "cyan" ? "border-hud-cyan/60" : "border-hud-green/60";

  return (
    <div className="border border-hud-border bg-[#090909] p-3">
      <span className="block text-[12px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-2">{title}</span>
      {paths.length === 0 ? (
        <span className="text-[12px] text-hud-dim">—</span>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-2">
          {paths.map((path, index) => {
            const src = frameUrl(path);
            const fileName = fileNameFromPath(path);
            return (
              <button
                key={`${path}-${index}`}
                type="button"
                onClick={() => onOpen(src, fileName)}
                className={`border ${border} bg-black p-1 text-left`}
                title={fileName}
              >
                <img
                  src={src}
                  alt={fileName}
                  className="h-24 w-full object-contain bg-[#040404]"
                  loading="lazy"
                />
                <span className="mt-1 block truncate text-[11px] text-hud-dim">{fileName}</span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default function ImageBenchmarks() {
  const [data, setData] = useState<ImageBenchmarkData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeAlbum, setActiveAlbum] = useState<string | null>(null);
  const [expandedPromptId, setExpandedPromptId] = useState<number | null>(null);
  const [lightbox, setLightbox] = useState<{ src: string; caption: string } | null>(null);

  useEffect(() => {
    fetchImageBenchmarks().then(setData).catch((e) => setError(e.message));
  }, []);

  const photobenchPrompts = useMemo(() => {
    if (!data) return [] as PhotobenchPromptView[];
    return data.prompts
      .map((prompt) => parsePhotobenchPrompt(prompt))
      .filter((item): item is PhotobenchPromptView => item !== null)
      .sort((a, b) => a.queryId.localeCompare(b.queryId, undefined, { numeric: true }));
  }, [data]);

  const byAlbum = useMemo(() => {
    const grouped = new Map<string, PhotobenchPromptView[]>();
    for (const prompt of photobenchPrompts) {
      if (!grouped.has(prompt.album)) grouped.set(prompt.album, []);
      grouped.get(prompt.album)!.push(prompt);
    }
    return grouped;
  }, [photobenchPrompts]);

  const albums = useMemo(() => Array.from(byAlbum.keys()).sort((a, b) => a.localeCompare(b, undefined, { numeric: true })), [byAlbum]);

  const currentAlbum = activeAlbum && byAlbum.has(activeAlbum) ? activeAlbum : (albums[0] ?? null);
  const activeQueries = currentAlbum ? byAlbum.get(currentAlbum) ?? [] : [];

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
        <span className="text-xs text-hud-dim uppercase tracking-[0.2em] animate-pulse">Loading image benchmarks...</span>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen flex-col">
      <div className="border-b border-hud-border bg-hud-panel px-6 py-3">
        <div className="flex items-center justify-end gap-8">
          <Stat label="QUERIES" value={String(photobenchPrompts.length)} />
          <Stat label="ALBUMS" value={String(albums.length)} />
          <Stat label="TOTAL COST" value={`$${data.summary.totalCostUsd.toFixed(2)}`} />
          <Stat label="TOTAL TIME" value={formatTime(data.summary.totalWallTimeS)} />
          <Link
            href="/images/run"
            className="flex items-center gap-1.5 border border-foreground px-3 py-1.5 text-[12px] font-bold uppercase tracking-[0.2em] text-foreground transition-colors hover:bg-foreground hover:text-background"
          >
            <Play size={12} />
            Live Run
          </Link>
        </div>
      </div>

      <div className="flex-1 p-4 space-y-4">
        <div className="border border-hud-border bg-hud-panel p-3">
          <span className="block text-[12px] font-bold uppercase tracking-[0.15em] text-hud-label mb-2">Album Folders</span>
          <div className="flex flex-wrap gap-2">
            {albums.map((album) => {
              const isActive = album === currentAlbum;
              return (
                <button
                  key={album}
                  type="button"
                  onClick={() => {
                    setActiveAlbum(album);
                    setExpandedPromptId(null);
                  }}
                  className={`border px-3 py-2 text-[12px] font-bold uppercase tracking-[0.15em] transition-colors ${
                    isActive
                      ? "border-hud-green bg-hud-green/10 text-hud-green"
                      : "border-hud-border text-hud-dim hover:text-foreground"
                  }`}
                >
                  {album} ({byAlbum.get(album)?.length ?? 0})
                </button>
              );
            })}
          </div>
        </div>

        <div className="border border-hud-border bg-hud-panel">
          <div className="border-b border-hud-border px-3 py-2">
            <span className="text-[12px] font-bold uppercase tracking-[0.15em] text-hud-label">
              {currentAlbum ?? "album"} / queries
            </span>
          </div>

          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-hud-border text-[12px] font-bold uppercase tracking-[0.15em] text-hud-dim">
                <th className="px-3 py-2 text-right w-10">#</th>
                <th className="px-3 py-2 text-left">Query</th>
                <th className="px-3 py-2 text-center w-28">Ground Truth</th>
                <th className="px-3 py-2 text-center w-28">Predictions</th>
                <th className="px-3 py-2 text-center w-24">Precision</th>
                <th className="px-3 py-2 text-center w-24">Recall</th>
                <th className="px-3 py-2 text-center w-24">Evidence</th>
                <th className="px-3 py-2 text-center w-24">Traces</th>
              </tr>
            </thead>
            <tbody>
              {activeQueries.map((entry, index) => {
                const isExpanded = expandedPromptId === entry.prompt.promptId;
                const traceCount = entry.result.traceEvents?.length ?? 0;
                return (
                  <FragmentRow
                    key={entry.prompt.promptId}
                    row={entry}
                    index={index + 1}
                    isExpanded={isExpanded}
                    traces={traceCount}
                    onToggle={() => setExpandedPromptId(isExpanded ? null : entry.prompt.promptId)}
                    onOpenLightbox={(src, caption) => setLightbox({ src, caption })}
                  />
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {lightbox && (
        <div className="fixed inset-0 z-[120] bg-black/85 backdrop-blur-sm p-4 flex items-center justify-center" onClick={() => setLightbox(null)}>
          <div className="relative max-w-[95vw] max-h-[95vh]" onClick={(e) => e.stopPropagation()}>
            <img
              src={lightbox.src}
              alt={lightbox.caption}
              className="max-w-[95vw] max-h-[88vh] object-contain border border-hud-border bg-black"
              loading="eager"
            />
            <div className="mt-2 text-[12px] text-hud-dim">{lightbox.caption}</div>
          </div>
        </div>
      )}
    </div>
  );
}

function FragmentRow({
  row,
  index,
  traces,
  isExpanded,
  onToggle,
  onOpenLightbox,
}: {
  row: PhotobenchPromptView;
  index: number;
  traces: number;
  isExpanded: boolean;
  onToggle: () => void;
  onOpenLightbox: (src: string, caption: string) => void;
}) {
  return (
    <>
      <tr
        onClick={onToggle}
        className={`border-b border-hud-border/50 cursor-pointer transition-colors ${isExpanded ? "bg-hud-border/20" : "hover:bg-hud-border/10"}`}
      >
        <td className="px-3 py-2 text-right tabular-nums text-hud-dim">{index}</td>
        <td className="px-3 py-2">
          <div className="font-semibold text-foreground">{row.result.question}</div>
          <div className="text-[11px] text-hud-dim mt-0.5">{row.queryId}</div>
        </td>
        <td className="px-3 py-2 text-center tabular-nums text-hud-green">{row.groundTruth.length}</td>
        <td className="px-3 py-2 text-center tabular-nums text-hud-cyan">{row.predictions.length}</td>
        <td className="px-3 py-2 text-center tabular-nums text-hud-cyan">{formatRatio(row.precision)}</td>
        <td className="px-3 py-2 text-center tabular-nums text-hud-green">{formatRatio(row.recall)}</td>
        <td className="px-3 py-2 text-center tabular-nums text-hud-dim">{row.evidenceLines.length}</td>
        <td className="px-3 py-2 text-center tabular-nums text-hud-dim">{traces}</td>
      </tr>

      {isExpanded && (
        <tr>
          <td colSpan={8} className="p-0 bg-[#080808] border-b border-hud-border/50">
            <div className="p-4 space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                <div className="border border-hud-border bg-[#0a0a0a] px-3 py-2">
                  <span className="block text-[12px] uppercase tracking-[0.12em] text-hud-dim">Ground Truth</span>
                  <span className="block mt-1 text-hud-green font-bold">{row.groundTruth.length}</span>
                </div>
                <div className="border border-hud-border bg-[#0a0a0a] px-3 py-2">
                  <span className="block text-[12px] uppercase tracking-[0.12em] text-hud-dim">Predictions</span>
                  <span className="block mt-1 text-hud-cyan font-bold">{row.predictions.length}</span>
                </div>
                <div className="border border-hud-border bg-[#0a0a0a] px-3 py-2">
                  <span className="block text-[12px] uppercase tracking-[0.12em] text-hud-dim">Precision</span>
                  <span className="block mt-1 text-hud-cyan font-bold">{formatRatio(row.precision)}</span>
                </div>
                <div className="border border-hud-border bg-[#0a0a0a] px-3 py-2">
                  <span className="block text-[12px] uppercase tracking-[0.12em] text-hud-dim">Recall</span>
                  <span className="block mt-1 text-hud-green font-bold">{formatRatio(row.recall)}</span>
                </div>
              </div>

              <ImageGrid title="Ground Truth Images" paths={row.groundTruthImagePaths} color="green" onOpen={onOpenLightbox} />
              <ImageGrid title="Prediction Images" paths={row.predictionImagePaths} color="cyan" onOpen={onOpenLightbox} />

              <div className="border border-hud-border bg-[#090909] p-3">
                <span className="block text-[12px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-2">Evidences</span>
                {row.evidenceLines.length === 0 ? (
                  <span className="text-[12px] text-hud-dim">—</span>
                ) : (
                  <ul className="space-y-1 list-disc pl-5 text-[12px] text-foreground/90">
                    {row.evidenceLines.map((line, i) => (
                      <li key={`${line.slice(0, 40)}-${i}`}>{line}</li>
                    ))}
                  </ul>
                )}
              </div>

              <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-[12px]">
                <Metric label="Iterations" value={String(row.result.iterations)} />
                <Metric label="Cost" value={`$${row.result.costUsd.toFixed(4)}`} />
                <Metric label="Wall Time" value={`${row.result.wallTimeS.toFixed(1)}s`} />
                <Metric label="Input Tokens" value={row.result.inputTokens.toLocaleString()} />
                <Metric label="Output Tokens" value={row.result.outputTokens.toLocaleString()} />
              </div>

              <div>
                <span className="block text-[12px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-2">Traces</span>
                {row.result.traceEvents && row.result.traceEvents.length > 0 ? (
                  <div className="h-80 border border-hud-border bg-[#090909]">
                    <TraceTimeline
                      events={row.result.traceEvents}
                      startTime={(row.result.traceEvents[0] as TraceEvent | undefined)?.timestamp ?? null}
                    />
                  </div>
                ) : (
                  <div className="border border-hud-border bg-[#090909] p-3 text-[12px] text-hud-dim">No trace events.</div>
                )}
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-hud-border bg-[#0a0a0a] px-3 py-2">
      <span className="block text-[11px] uppercase tracking-[0.12em] text-hud-dim">{label}</span>
      <span className="block mt-0.5 tabular-nums text-foreground">{value}</span>
    </div>
  );
}
