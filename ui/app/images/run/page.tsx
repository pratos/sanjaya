"use client";

import { useState, useRef } from "react";
import { Upload, X, ImageIcon } from "lucide-react";
import { useImageRun } from "@/lib/use-image-run";
import { uploadImages } from "@/lib/api";

import { RunStatusHeader } from "@/components/hud/run-status";
import { TraceTimeline } from "@/components/hud/trace-timeline";
import { FinalAnswerPanel } from "@/components/hud/final-answer-panel";
import { Panel } from "@/components/hud/panel";

const ACCEPTED_TYPES = ".jpg,.jpeg,.png,.webp,.gif,.tiff,.tif,.bmp,.heic,.heif,.svg";

export default function ImageRunPage() {
  const { state, startRun, reset, tokenTotals } = useImageRun();

  const [files, setFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<Record<string, string>>({});
  const [uploadedPaths, setUploadedPaths] = useState<string[]>([]);
  const [question, setQuestion] = useState("");
  const [maxIterations, setMaxIterations] = useState(10);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const isRunning = state.status === "running";
  const disabled = isRunning || uploading;

  const wallTimeS = state.startTime
    ? (Date.now() / 1000 - state.startTime)
    : 0;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newFiles = Array.from(e.target.files ?? []);
    setFiles((prev) => [...prev, ...newFiles]);
    setUploadError(null);

    // Generate previews
    for (const file of newFiles) {
      const reader = new FileReader();
      reader.onload = (ev) => {
        setPreviews((prev) => ({ ...prev, [file.name]: ev.target?.result as string }));
      };
      reader.readAsDataURL(file);
    }

    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const removeFile = (index: number) => {
    const removed = files[index];
    setFiles((prev) => prev.filter((_, i) => i !== index));
    if (removed) {
      setPreviews((prev) => {
        const next = { ...prev };
        delete next[removed.name];
        return next;
      });
    }
  };

  const handleSubmit = async () => {
    if (files.length === 0 || !question.trim()) return;

    setUploading(true);
    setUploadError(null);

    try {
      const { paths } = await uploadImages(files);
      setUploadedPaths(paths);

      await startRun({
        imagePaths: paths,
        question: question.trim(),
        maxIterations,
      });
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  };

  const handleReset = () => {
    reset();
    setFiles([]);
    setPreviews({});
    setUploadedPaths([]);
    setUploadError(null);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      handleSubmit();
    }
  };

  const labelClass = "block text-xs font-bold uppercase tracking-[0.15em] text-hud-dim mb-1";
  const inputClass = "w-full border border-hud-border bg-[#0a0a0a] px-2 py-2 text-sm text-foreground placeholder:text-hud-dim focus:border-hud-border-active focus:outline-none disabled:opacity-40";

  return (
    <div className="flex min-h-screen flex-col">
      <RunStatusHeader
        status={state.status}
        runId={state.runId}
        iteration={state.currentIteration}
        maxIterations={state.maxIterations}
        startTime={state.startTime}
      />

      <div className="flex-1 grid grid-cols-[minmax(320px,400px)_1fr] gap-0 border-t border-hud-border">
        {/* Left panel — Image upload + query */}
        <div className="border-r border-hud-border overflow-y-auto">
          <Panel title="IMAGE QUERY">
            <div className="space-y-3" onKeyDown={handleKeyDown}>
              {/* File upload */}
              <div>
                <label className={labelClass}>IMAGES</label>
                <div
                  onClick={() => !disabled && fileInputRef.current?.click()}
                  className={`border border-dashed border-hud-border bg-[#0a0a0a] px-4 py-4 text-center cursor-pointer transition-colors hover:border-hud-border-active ${disabled ? "opacity-40 cursor-not-allowed" : ""}`}
                >
                  <Upload size={18} className="mx-auto text-hud-dim mb-1" />
                  <p className="text-xs text-hud-dim">
                    Click to add images (JPG, PNG, WebP, GIF, TIFF, BMP, HEIC, SVG)
                  </p>
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept={ACCEPTED_TYPES}
                  multiple
                  onChange={handleFileChange}
                  className="hidden"
                />

                {/* File list with thumbnails */}
                {files.length > 0 && (
                  <div className="mt-2 space-y-1">
                    {files.map((file, i) => (
                      <div
                        key={`${file.name}-${i}`}
                        className="flex items-center gap-2 border border-hud-border px-2 py-1.5 text-xs"
                      >
                        {previews[file.name] ? (
                          <img
                            src={previews[file.name]}
                            alt={file.name}
                            className="w-8 h-8 object-cover shrink-0"
                          />
                        ) : (
                          <ImageIcon size={14} className="text-hud-cyan shrink-0" />
                        )}
                        <span className="flex-1 truncate text-foreground">{file.name}</span>
                        <span className="text-hud-dim text-[12px] tabular-nums shrink-0">
                          {(file.size / 1024).toFixed(0)}K
                        </span>
                        <button
                          onClick={(e) => { e.stopPropagation(); removeFile(i); }}
                          disabled={disabled}
                          className="text-hud-dim hover:text-hud-red transition-colors disabled:opacity-30"
                        >
                          <X size={14} />
                        </button>
                      </div>
                    ))}
                  </div>
                )}

                {uploadError && (
                  <p className="mt-1 text-xs text-hud-red">{uploadError}</p>
                )}
              </div>

              {/* Question */}
              <div>
                <label className={labelClass}>QUESTION</label>
                <input
                  type="text"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask a question about the images..."
                  disabled={disabled}
                  className={inputClass}
                />
              </div>

              {/* Max iterations */}
              <div>
                <label className={labelClass}>MAX ITERATIONS</label>
                <input
                  type="number"
                  min={1}
                  max={20}
                  value={maxIterations}
                  onChange={(e) => setMaxIterations(Number(e.target.value) || 10)}
                  disabled={disabled}
                  className={inputClass}
                />
              </div>

              {/* Execute / Reset */}
              <div className="flex gap-2">
                <button
                  onClick={handleSubmit}
                  disabled={disabled || files.length === 0 || !question.trim()}
                  className="flex-1 border border-foreground bg-transparent px-3 py-2 text-xs font-bold uppercase tracking-[0.2em] text-foreground transition-colors hover:bg-foreground hover:text-background disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:bg-transparent disabled:hover:text-foreground"
                >
                  {uploading ? "UPLOADING..." : isRunning ? "RUNNING..." : "EXECUTE"}
                </button>
                {state.status !== "idle" && (
                  <button
                    onClick={handleReset}
                    disabled={isRunning}
                    className="border border-hud-border px-3 py-2 text-xs font-bold uppercase tracking-[0.15em] text-hud-dim transition-colors hover:border-hud-border-active hover:text-foreground disabled:opacity-30"
                  >
                    RESET
                  </button>
                )}
              </div>

              {uploadedPaths.length > 0 && (
                <div className="text-[12px] text-hud-dim">
                  <span className="uppercase tracking-wider">Uploaded:</span>{" "}
                  {uploadedPaths.length} image(s)
                </div>
              )}
            </div>
          </Panel>

          {/* Session metrics */}
          {state.status !== "idle" && (
            <Panel title="METRICS">
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-hud-dim">Iteration</span>
                  <span className="tabular-nums">{state.currentIteration}/{state.maxIterations}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-hud-dim">Tokens</span>
                  <span className="tabular-nums">{(tokenTotals.inputTokens / 1000).toFixed(1)}K in / {(tokenTotals.outputTokens / 1000).toFixed(1)}K out</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-hud-dim">Cost</span>
                  <span className="tabular-nums text-hud-amber">${tokenTotals.costUsd.toFixed(4)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-hud-dim">Wall Time</span>
                  <span className="tabular-nums">{wallTimeS.toFixed(0)}s</span>
                </div>
              </div>
            </Panel>
          )}
        </div>

        {/* Right panel — Timeline + Answer */}
        <div className="overflow-y-auto p-4 space-y-4">
          {state.status === "idle" && (
            <div className="flex items-center justify-center h-full">
              <span className="text-sm text-hud-dim uppercase tracking-[0.2em]">
                Upload images and ask a question to begin
              </span>
            </div>
          )}

          {state.events.length > 0 && (
            <div className="h-72">
              <TraceTimeline
                events={state.events}
                startTime={state.startTime}
              />
            </div>
          )}

          {(state.status === "complete" || state.status === "error") && (
            <FinalAnswerPanel
              status={state.status}
              finalAnswer={state.finalAnswer}
              finalStatus={state.finalStatus}
              question={question}
              events={state.events}
              iterations={state.currentIteration}
              costUsd={tokenTotals.costUsd}
              wallTimeS={wallTimeS}
            />
          )}

          {state.status === "error" && state.error && !state.finalAnswer && (
            <div className="border border-hud-red bg-hud-red/5 px-4 py-3">
              <span className="text-xs text-hud-red">{state.error}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
