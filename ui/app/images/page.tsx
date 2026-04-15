"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Play } from "lucide-react";
import type { ImageBenchmarkData, ImagePrompt, ImageResult, TraceEvent } from "@/lib/types";
import { fetchImageBenchmarks, frameUrl } from "@/lib/api";
import { AnswerRenderer } from "@/components/benchmark/answer-renderer";
import { TraceTimeline } from "@/components/hud/trace-timeline";

function formatTime(seconds: number): string {
  if (seconds >= 60) return `${(seconds / 60).toFixed(1)}m`;
  return `${seconds.toFixed(0)}s`;
}

function formatName(name: string): string {
  return name.replace(/_/g, " ");
}

function scoreBadge(result: ImageResult | undefined) {
  if (!result || result.error) {
    return <span className="text-hud-red text-[12px]">ERROR</span>;
  }

  // Show correctness verdict if available from benchmark data
  const correct = result.answerData?.correct;
  if (typeof correct === "boolean") {
    return (
      <span className="flex items-center gap-1.5">
        <span className={`text-[12px] font-bold ${correct ? "text-hud-green" : "text-hud-red"}`}>
          {correct ? "✓" : "✗"}
        </span>
        <span className="text-[12px] text-hud-dim">{result.iterations}it</span>
      </span>
    );
  }

  const iters = result.iterations;
  if (iters >= 10) {
    return <span className="text-hud-amber text-[12px]">FORCED</span>;
  }
  if (iters <= 3) {
    return <span className="text-hud-green text-[12px] font-bold">{iters} iters</span>;
  }
  if (iters <= 6) {
    return <span className="text-hud-green text-[12px]">{iters} iters</span>;
  }
  return <span className="text-hud-amber text-[12px]">{iters} iters</span>;
}

function Stat({ label, value, color, small }: { label: string; value: string; color?: string; small?: boolean }) {
  return (
    <div className="flex flex-col items-end gap-0.5">
      <span className="text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim">{label}</span>
      <span className={`${small ? "text-[12px]" : "text-xs"} font-bold tabular-nums ${color ?? "text-foreground"}`}>{value}</span>
    </div>
  );
}

function MetricRow({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-[13px] uppercase tracking-wider text-hud-dim">{label}</span>
      <span className={`text-[12px] tabular-nums ${color ?? "text-foreground"}`}>{value}</span>
    </div>
  );
}

function VerdictStat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="border border-hud-border bg-[#0a0a0a] px-3 py-2">
      <span className="block text-[12px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-1">{label}</span>
      <span className={`block text-sm font-bold ${color ?? "text-foreground"}`}>{value}</span>
    </div>
  );
}

function decodePathPart(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function fileNameFromPath(path: string): string {
  const raw = path.split("/").filter(Boolean).pop() ?? path;
  return decodePathPart(raw);
}

function fileStemFromPath(path: string): string {
  return fileNameFromPath(path).replace(/\.[^.]+$/, "");
}

function parseChoiceLabel(value: string | null): string | null {
  if (!value) return null;
  const trimmed = value.trim().toUpperCase();
  const direct = trimmed.match(/^\(?([A-Z])\)?$/);
  if (direct) return direct[1];
  return null;
}

function normalizePath(value: string | null): string {
  if (!value) return "";
  const stripped = value.startsWith("file://") ? value.slice("file://".length) : value;
  return decodePathPart(stripped);
}

function collectStrings(value: unknown): string[] {
  if (typeof value === "string") return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string");
}

function normalizeComparableToken(value: string): string {
  return value.trim().toLowerCase().replace(/\.[a-z0-9]+$/i, "");
}

interface IterationModelSummary {
  iteration: number;
  models: string[];
  costUsd: number;
}

function summarizeIterationModels(events: TraceEvent[] | null): IterationModelSummary[] {
  if (!events || events.length === 0) return [];

  const usageKinds = new Set(["root_response", "sub_llm", "vision", "schema_generation", "critic_evaluation"]);
  const byIteration = new Map<number, { models: Set<string>; costUsd: number }>();

  for (const event of events) {
    if (!usageKinds.has(event.kind)) continue;

    const payload = event.payload ?? {};
    const iterRaw = payload["iteration"];
    const iteration = typeof iterRaw === "number" ? iterRaw : 0;

    const modelRaw = payload["model"] ?? payload["model_used"] ?? payload["orchestrator_model"] ?? payload["recursive_model"];
    const model = typeof modelRaw === "string" && modelRaw.trim().length > 0 ? modelRaw : null;

    const costRaw = payload["cost_usd"];
    const cost = typeof costRaw === "number" ? costRaw : 0;

    if (!byIteration.has(iteration)) {
      byIteration.set(iteration, { models: new Set<string>(), costUsd: 0 });
    }

    const bucket = byIteration.get(iteration)!;
    if (model) bucket.models.add(model);
    bucket.costUsd += cost;
  }

  return Array.from(byIteration.entries())
    .map(([iteration, value]) => ({
      iteration,
      models: Array.from(value.models).sort(),
      costUsd: value.costUsd,
    }))
    .sort((a, b) => a.iteration - b.iteration);
}

function ImageDetail({ prompt }: { prompt: ImagePrompt }) {
  const versionKeys = Object.keys(prompt.versions).sort();
  const [activeVersion, setActiveVersion] = useState(prompt.bestVersion);
  const active = prompt.versions[activeVersion];
  const traceEvents: TraceEvent[] | null = active?.traceEvents ?? null;
  const iterationModelRows = summarizeIterationModels(traceEvents);
  const [lightbox, setLightbox] = useState<{
    src: string;
    alt: string;
    caption: string;
  } | null>(null);

  useEffect(() => {
    if (!lightbox) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setLightbox(null);
      }
    };

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousOverflow;
    };
  }, [lightbox]);

  const answerData = active?.answerData ?? null;

  const predictedValues =
    collectStrings(answerData?.["predicted"]).length > 0
      ? collectStrings(answerData?.["predicted"])
      : collectStrings(answerData?.["prediction"]).length > 0
        ? collectStrings(answerData?.["prediction"])
        : collectStrings(answerData?.["predictions"]).length > 0
          ? collectStrings(answerData?.["predictions"])
          : collectStrings(answerData?.["matching_filenames"]).length > 0
            ? collectStrings(answerData?.["matching_filenames"])
            : collectStrings(answerData?.["answer"]);

  const groundTruthValues =
    collectStrings(answerData?.["ground_truth"]).length > 0
      ? collectStrings(answerData?.["ground_truth"])
      : collectStrings(answerData?.["groundTruth"]).length > 0
        ? collectStrings(answerData?.["groundTruth"])
        : active?.groundTruth
          ? [active.groundTruth]
          : [];

  const predictedRaw: string | null = predictedValues.length > 0 ? predictedValues.join(", ") : null;
  const answerGroundTruthRaw: string | null = groundTruthValues.length > 0 ? groundTruthValues.join(", ") : null;

  const correctnessRaw = answerData?.["correct"];
  const metrics =
    answerData?.["metrics"] && typeof answerData["metrics"] === "object"
      ? (answerData["metrics"] as Record<string, unknown>)
      : null;

  const derivedCorrectness = (() => {
    if (typeof correctnessRaw === "boolean") return correctnessRaw;

    if (predictedValues.length > 0 && groundTruthValues.length > 0) {
      const predictedSet = new Set(predictedValues.map(normalizeComparableToken));
      const truthSet = new Set(groundTruthValues.map(normalizeComparableToken));

      if (predictedSet.size !== truthSet.size) return false;
      for (const item of predictedSet) {
        if (!truthSet.has(item)) return false;
      }
      return true;
    }

    const tp = typeof metrics?.["tp"] === "number" ? metrics["tp"] : null;
    const fp = typeof metrics?.["fp"] === "number" ? metrics["fp"] : null;
    const fn = typeof metrics?.["fn"] === "number" ? metrics["fn"] : null;
    if (tp !== null && fp !== null && fn !== null) {
      return fp === 0 && fn === 0 && tp > 0;
    }

    return null;
  })();

  const isCorrect: boolean | null = typeof derivedCorrectness === "boolean" ? derivedCorrectness : null;

  const benchmark = typeof answerData?.["benchmark"] === "string" ? answerData["benchmark"] : null;
  const muirbench = active?.muirbench ?? null;
  const isMuirbench = benchmark === "MUIRBench" || !!muirbench;

  const predictedLabel = parseChoiceLabel(predictedValues[0] ?? null);
  const parsedGroundTruthLabel = parseChoiceLabel(groundTruthValues[0] ?? null);
  const manifestGroundTruthLabel = parseChoiceLabel(muirbench?.answer ?? null);
  const groundTruthLabel = parsedGroundTruthLabel ?? manifestGroundTruthLabel;

  const predictedChoice = predictedLabel
    ? muirbench?.choices.find((choice) => choice.label === predictedLabel) ?? null
    : null;
  const groundTruthChoice = groundTruthLabel
    ? muirbench?.choices.find((choice) => choice.label === groundTruthLabel) ?? null
    : null;

  const formatMuirbenchValue = (
    label: string | null,
    choice: { text: string; isImageRef: boolean; imageTag: string | null } | null
  ): string => {
    if (!label) return "—";
    if (!choice) return label;
    if (!choice.isImageRef) return choice.text || label;
    return choice.imageTag || label;
  };

  const predictedDisplay = isMuirbench
    ? formatMuirbenchValue(predictedLabel, predictedChoice)
    : predictedRaw;

  const groundTruthDisplay = isMuirbench
    ? formatMuirbenchValue(groundTruthLabel, groundTruthChoice)
    : answerGroundTruthRaw;

  const answerText = active?.answerText ?? "";
  const answerLooksTruncated = answerText.includes("[truncated]") || answerText.length === 500;

  const referenceTag = active?.imagePaths[0] ? fileStemFromPath(active.imagePaths[0]) : null;

  const imagePathByToken = new Map<string, string>();
  for (const path of active.imagePaths) {
    const fileName = fileNameFromPath(path);
    const fileStem = fileStemFromPath(path);
    imagePathByToken.set(normalizeComparableToken(fileName), path);
    imagePathByToken.set(normalizeComparableToken(fileStem), path);
  }

  const resolveResultImages = (values: string[]) => {
    const seen = new Set<string>();
    const resolved: Array<{ token: string; path: string }> = [];

    for (const token of values) {
      const mapped = imagePathByToken.get(normalizeComparableToken(token));
      if (!mapped || seen.has(mapped)) continue;
      seen.add(mapped);
      resolved.push({ token, path: mapped });
    }

    return resolved;
  };

  const predictedResolvedImages = resolveResultImages(predictedValues);
  const groundTruthResolvedImages = resolveResultImages(groundTruthValues);

  if (!active) return null;

  return (
    <>
      <div className="bg-[#080808] border-t border-hud-border">
      {versionKeys.length > 1 && (
        <div className="flex border-b border-hud-border">
          {versionKeys.map((v) => (
            <button
              key={v}
              onClick={() => setActiveVersion(v)}
              className={`px-4 py-1.5 text-[12px] font-bold uppercase tracking-[0.15em] transition-colors ${
                v === activeVersion ? "bg-hud-panel text-foreground border-b-2 border-hud-green" : "text-hud-dim hover:text-foreground"
              }`}
            >
              {v.toUpperCase()}

            </button>
          ))}
        </div>
      )}

      <div className="p-4 space-y-4">
        <div>
          <span className="block text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-1">Question</span>
          <p className="text-[13px] text-hud-label leading-relaxed whitespace-pre-wrap">{prompt.question}</p>
        </div>

        {(predictedDisplay || groundTruthDisplay || isCorrect !== null) && (
          <div>
            <span className="block text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-2">Result</span>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
              <VerdictStat label="Predicted" value={predictedDisplay ? String(predictedDisplay) : "—"} color="text-hud-cyan" />
              <VerdictStat label="Ground Truth" value={groundTruthDisplay ? String(groundTruthDisplay) : "—"} color="text-hud-green" />
              <VerdictStat
                label="Match"
                value={isCorrect === null ? "—" : isCorrect ? "CORRECT" : "INCORRECT"}
                color={isCorrect === null ? "text-foreground" : isCorrect ? "text-hud-green" : "text-hud-red"}
              />
            </div>

            {(predictedResolvedImages.length > 0 || groundTruthResolvedImages.length > 0) && (
              <div className="mt-2 grid grid-cols-1 sm:grid-cols-2 gap-2">
                {predictedResolvedImages.length > 0 && (
                  <div className="border border-hud-border bg-[#0a0a0a] px-3 py-2">
                    <span className="block text-[12px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-2">Predicted Image</span>
                    <div className="flex flex-wrap gap-2">
                      {predictedResolvedImages.map((item, i) => {
                        const src = frameUrl(item.path);
                        const fileName = fileNameFromPath(item.path);
                        return (
                          <button
                            key={`${item.path}-${i}`}
                            type="button"
                            onClick={() =>
                              setLightbox({
                                src,
                                alt: `predicted image ${i + 1}`,
                                caption: `Predicted • ${fileName}`,
                              })
                            }
                            className="border border-hud-cyan/50 bg-black p-1"
                            title={fileName}
                          >
                            <img
                              src={src}
                              alt={`predicted image ${i + 1}`}
                              className="h-20 w-20 object-contain bg-[#050505]"
                              loading="lazy"
                            />
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}

                {groundTruthResolvedImages.length > 0 && (
                  <div className="border border-hud-border bg-[#0a0a0a] px-3 py-2">
                    <span className="block text-[12px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-2">Ground Truth Image</span>
                    <div className="flex flex-wrap gap-2">
                      {groundTruthResolvedImages.map((item, i) => {
                        const src = frameUrl(item.path);
                        const fileName = fileNameFromPath(item.path);
                        return (
                          <button
                            key={`${item.path}-${i}`}
                            type="button"
                            onClick={() =>
                              setLightbox({
                                src,
                                alt: `ground truth image ${i + 1}`,
                                caption: `Ground Truth • ${fileName}`,
                              })
                            }
                            className="border border-hud-green/50 bg-black p-1"
                            title={fileName}
                          >
                            <img
                              src={src}
                              alt={`ground truth image ${i + 1}`}
                              className="h-20 w-20 object-contain bg-[#050505]"
                              loading="lazy"
                            />
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {isMuirbench && (
          <div>
            <span className="block text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-2">MUIRBench Layout</span>

            {referenceTag && (
              <div className="mb-2 border border-hud-border bg-[#0a0a0a] px-3 py-2">
                <span className="block text-[12px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-1">Reference</span>
                <span className="text-sm font-bold text-foreground">{referenceTag}</span>
              </div>
            )}

            {muirbench && muirbench.choices.length > 0 && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {muirbench.choices.map((choice) => {
                  const isPred = choice.label === predictedLabel;
                  const isGT = choice.label === groundTruthLabel;
                  const displayValue = choice.isImageRef
                    ? choice.imageTag ?? "<image>"
                    : choice.text;

                  return (
                    <div
                      key={choice.label}
                      className={`border px-3 py-2 bg-[#0a0a0a] ${
                        isGT ? "border-hud-green/60" : isPred ? "border-hud-cyan/60" : "border-hud-border"
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-[12px] font-bold uppercase tracking-[0.15em] text-hud-dim">
                          Candidate {choice.label}
                        </span>
                        <div className="flex items-center gap-1.5">
                          {isPred && <span className="text-[11px] text-hud-cyan">Predicted</span>}
                          {isGT && <span className="text-[11px] text-hud-green">Ground Truth</span>}
                        </div>
                      </div>
                      <span className="block text-sm font-bold text-foreground mt-1">{displayValue || "—"}</span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {active.imagePaths.length > 0 && (
          <div>
            <span className="block text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-2">Images</span>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {active.imagePaths.map((path, index) => {
                const src = frameUrl(path);
                const fileName = fileNameFromPath(path);

                const choiceLabels = (muirbench?.choices ?? [])
                  .filter((choice) => choice.imagePath && normalizePath(choice.imagePath) === normalizePath(path))
                  .map((choice) => choice.label);

                const isPredictedImage = choiceLabels.some((label) => label === predictedLabel);
                const isGroundTruthImage = choiceLabels.some((label) => label === groundTruthLabel);

                return (
                  <div key={`${path}-${index}`} className="space-y-1">
                    <button
                      type="button"
                      onClick={() =>
                        setLightbox({
                          src,
                          alt: `input image ${index + 1}`,
                          caption: `img_${index} • ${fileName}`,
                        })
                      }
                      className={`block w-full border bg-black p-1 transition-colors hover:border-hud-border-active ${
                        isGroundTruthImage ? "border-hud-green/60" : isPredictedImage ? "border-hud-cyan/60" : "border-hud-border/70"
                      }`}
                    >
                      <img
                        src={src}
                        alt={`input image ${index + 1}`}
                        className="h-44 w-full object-contain bg-[#050505]"
                        loading="lazy"
                      />
                    </button>
                    <div className="space-y-0.5">
                      <span title={`img_${index} • ${fileName}`} className="block truncate text-[12px] text-hud-dim">
                        {`img_${index} • ${fileName}`}
                      </span>
                      {(index === 0 || choiceLabels.length > 0 || isPredictedImage || isGroundTruthImage) && (
                        <div className="flex flex-wrap items-center gap-1">
                          {index === 0 && <span className="text-[11px] px-1.5 py-0.5 border border-hud-border text-hud-dim">Reference</span>}
                          {choiceLabels.map((label) => (
                            <span key={label} className="text-[11px] px-1.5 py-0.5 border border-hud-border text-hud-dim">
                              {`Candidate ${label}`}
                            </span>
                          ))}
                          {isPredictedImage && (
                            <span className="text-[11px] px-1.5 py-0.5 border border-hud-cyan/50 text-hud-cyan">
                              Predicted
                            </span>
                          )}
                          {isGroundTruthImage && (
                            <span className="text-[11px] px-1.5 py-0.5 border border-hud-green/50 text-hud-green">
                              Ground Truth
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {answerText && (
          <div>
            <span className="block text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-1">Answer (Full Text)</span>
            <pre className="max-h-96 overflow-auto border border-hud-border bg-[#0a0a0a] p-3 text-[12px] text-foreground leading-relaxed whitespace-pre-wrap break-words">
              {answerText}
            </pre>
            {answerLooksTruncated && (
              <p className="mt-2 text-[12px] text-hud-amber">
                This stored answer appears truncated at source. Re-run or export full traces to inspect the complete model output.
              </p>
            )}
          </div>
        )}

        {active.groundTruth && !isMuirbench && (
          <div>
            <span className="block text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-1">Ground Truth</span>
            <p className="text-xs text-hud-green leading-relaxed">{active.groundTruth}</p>
          </div>
        )}

        {active.evidenceSources.length > 0 && (
          <div>
            <span className="block text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-2">Sources</span>
            <div className="flex flex-wrap gap-1.5">
              {active.evidenceSources.map((src) => (
                <span key={src} className="text-[12px] border border-hud-cyan/30 text-hud-cyan px-2 py-0.5">
                  {src.replace("image:", "")}
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="grid grid-cols-[200px_1fr] gap-4">
          <div className="space-y-1.5 border-r border-hud-border pr-4">
            <span className="block text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-2">Metrics</span>
            <MetricRow label="Iterations" value={String(active.iterations)} />
            <MetricRow label="Cost" value={`$${active.costUsd.toFixed(4)}`} color="text-hud-amber" />
            <MetricRow label="Tokens" value={`${(active.inputTokens / 1000).toFixed(1)}K / ${(active.outputTokens / 1000).toFixed(1)}K`} />
            <MetricRow label="Wall Time" value={`${active.wallTimeS.toFixed(1)}s`} />
            <MetricRow label="Evidence" value={String(active.evidenceCount)} />
            <MetricRow label="Images" value={String(active.imagePaths.length)} />
          </div>

          <div>
            <span className="block text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-2">Structured Data</span>
            <AnswerRenderer data={active.answerData} onSeek={() => {}} />
            {active.answerData && (
              <details className="mt-3 border border-hud-border bg-[#0a0a0a]">
                <summary className="cursor-pointer px-3 py-2 text-[12px] uppercase tracking-[0.12em] text-hud-dim">
                  Raw JSON
                </summary>
                <pre className="max-h-72 overflow-auto border-t border-hud-border p-3 text-[12px] text-foreground whitespace-pre-wrap break-words">
                  {JSON.stringify(active.answerData, null, 2)}
                </pre>
              </details>
            )}
          </div>
        </div>

        {iterationModelRows.length > 0 && (
          <div>
            <span className="block text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-2">Iteration Models</span>
            <div className="border border-hud-border bg-[#0a0a0a] overflow-x-auto">
              <table className="w-full text-[12px]">
                <thead>
                  <tr className="border-b border-hud-border text-hud-dim uppercase tracking-[0.12em]">
                    <th className="px-3 py-2 text-left">Iteration</th>
                    <th className="px-3 py-2 text-left">Models</th>
                    <th className="px-3 py-2 text-right">Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {iterationModelRows.map((row) => (
                    <tr key={row.iteration} className="border-b border-hud-border/50 last:border-b-0">
                      <td className="px-3 py-2 tabular-nums">{row.iteration}</td>
                      <td className="px-3 py-2">
                        {row.models.length > 0 ? row.models.join(" • ") : <span className="text-hud-dim">—</span>}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums text-hud-amber">${row.costUsd.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {traceEvents && traceEvents.length > 0 && (
          <div className="h-80">
            <TraceTimeline events={traceEvents} startTime={traceEvents[0]?.timestamp ?? null} />
          </div>
        )}
      </div>
    </div>

      {lightbox && (
        <div
          className="fixed inset-0 z-[120] bg-black/85 backdrop-blur-sm p-4 flex items-center justify-center"
          onClick={() => setLightbox(null)}
        >
          <div className="relative max-w-[95vw] max-h-[95vh]" onClick={(e) => e.stopPropagation()}>
            <img
              src={lightbox.src}
              alt={lightbox.alt}
              className="max-w-[95vw] max-h-[88vh] object-contain border border-hud-border bg-black"
              loading="eager"
            />
            <div className="mt-2 flex items-center justify-between gap-3 text-[12px]">
              <span className="truncate text-hud-dim" title={lightbox.caption}>{lightbox.caption}</span>
              <span className="text-hud-dim/80 uppercase tracking-[0.12em]">ESC to close</span>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default function ImageBenchmarks() {
  const [data, setData] = useState<ImageBenchmarkData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);

  useEffect(() => {
    fetchImageBenchmarks().then(setData).catch((e) => setError(e.message));
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
          Loading image benchmarks...
        </span>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen flex-col">
      <div className="border-b border-hud-border bg-hud-panel px-6 py-3">
        <div className="flex items-center justify-end">
          <div className="flex items-center gap-8">
            <Stat label="PROMPTS" value={String(data.summary.totalPrompts)} />
            <Stat label="VERSIONS" value={data.summary.versions.join(", ")} />
            <Stat label="TOTAL COST" value={`$${data.summary.totalCostUsd.toFixed(2)}`} color="text-hud-amber" />
            <Stat label="TOTAL TIME" value={formatTime(data.summary.totalWallTimeS)} />
            <div className="h-6 w-px bg-hud-border" />
            <Stat label="MODELS" value={data.summary.models?.join(" / ") ?? "—"} small />
            {data.summary.benchmarkSummaries && Object.entries(data.summary.benchmarkSummaries).map(([bm, acc]) => (
              <Stat
                key={bm}
                label={bm.toUpperCase()}
                value={`${acc.correct}/${acc.total} (${(acc.accuracy * 100).toFixed(0)}%)`}
                color={acc.accuracy >= 0.5 ? "text-hud-green" : acc.accuracy > 0 ? "text-hud-amber" : "text-hud-red"}
              />
            ))}
            <div className="h-6 w-px bg-hud-border" />
            <Link
              href="/images/run"
              className="flex items-center gap-1.5 border border-foreground px-3 py-1.5 text-[12px] font-bold uppercase tracking-[0.2em] text-foreground transition-colors hover:bg-foreground hover:text-background"
            >
              <Play size={12} />
              Live Run
            </Link>
          </div>
        </div>
      </div>

      <div className="flex-1 p-4 space-y-4">
        <div className="border border-hud-border bg-hud-panel">
          <div className="border-b border-hud-border px-3 py-2">
            <span className="text-[12px] font-bold uppercase tracking-[0.15em] text-hud-label">
              Image Analysis Results
            </span>
          </div>

          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-hud-border text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim">
                <th className="px-3 py-2 text-right w-8">#</th>
                <th className="px-3 py-2 text-left">Prompt</th>
                <th className="px-3 py-2 text-center w-20">Score</th>
                <th className="px-3 py-2 text-right w-16">Cost</th>
                <th className="px-3 py-2 text-right w-16">Time</th>
                <th className="px-3 py-2 text-right w-16">Tokens</th>
                <th className="px-3 py-2 text-center w-16">Evidence</th>
                <th className="px-3 py-2 text-center w-16">Images</th>
              </tr>
            </thead>
            <tbody>
              {data.prompts.map((prompt) => {
                const best = prompt.versions[prompt.bestVersion];
                const isExpanded = expandedId === prompt.promptId;
                return (
                  <PromptRow
                    key={prompt.promptId}
                    prompt={prompt}
                    best={best}
                    isExpanded={isExpanded}
                    onToggle={() => setExpandedId(isExpanded ? null : prompt.promptId)}
                  />
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function PromptRow({
  prompt,
  best,
  isExpanded,
  onToggle,
}: {
  prompt: ImagePrompt;
  best: ImageResult | undefined;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  return (
    <>
      <tr
        onClick={onToggle}
        className={`border-b border-hud-border/50 cursor-pointer transition-colors ${
          isExpanded ? "bg-hud-border/20" : "hover:bg-hud-border/10"
        }`}
      >
        <td className="px-3 py-2 text-right tabular-nums text-hud-dim">{prompt.promptId}</td>
        <td className="px-3 py-2 font-bold text-foreground">{formatName(prompt.promptName)}</td>
        <td className="px-3 py-2 text-center">{scoreBadge(best)}</td>
        <td className="px-3 py-2 text-right tabular-nums text-hud-amber">
          {best && !best.error ? `$${best.costUsd.toFixed(2)}` : "\u2014"}
        </td>
        <td className="px-3 py-2 text-right tabular-nums text-hud-dim">
          {best && !best.error ? formatTime(best.wallTimeS) : "\u2014"}
        </td>
        <td className="px-3 py-2 text-right tabular-nums text-hud-dim">
          {best && !best.error ? `${(best.inputTokens / 1000).toFixed(0)}K` : "\u2014"}
        </td>
        <td className="px-3 py-2 text-center tabular-nums text-hud-dim">
          {best && !best.error ? best.evidenceCount : "\u2014"}
        </td>
        <td className="px-3 py-2 text-center tabular-nums text-hud-dim">
          {best && !best.error ? best.imagePaths.length : "\u2014"}
        </td>
      </tr>
      {isExpanded && (
        <tr>
          <td colSpan={8} className="p-0">
            <ImageDetail prompt={prompt} />
          </td>
        </tr>
      )}
    </>
  );
}
