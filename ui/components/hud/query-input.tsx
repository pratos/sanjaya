"use client";

import { useEffect, useState } from "react";
import { Eye } from "lucide-react";
import { Panel } from "./panel";
import { VideoPreviewModal } from "./video-preview-modal";
import { fetchVideos, type VideoEntry } from "@/lib/api";

const SUBTITLE_MODES = ["none", "auto", "local", "api"] as const;
const SUBTITLE_API_MODELS = [
  "gpt-4o-transcribe-diarize",
  "whisper-1",
  "gpt-4o-mini-transcribe",
] as const;

interface RunParams {
  videoPath: string;
  question: string;
  subtitleMode: string;
  subtitleApiModel: string;
  maxIterations: number;
}

interface QueryInputProps {
  onSubmit: (params: RunParams) => void;
  onVideoChange?: (videoPath: string) => void;
  disabled: boolean;
}

export function QueryInput({ onSubmit, onVideoChange, disabled }: QueryInputProps) {
  const [videos, setVideos] = useState<VideoEntry[]>([]);
  const [videoPath, setVideoPath] = useState("");
  const [question, setQuestion] = useState("");
  const [subtitleMode, setSubtitleMode] = useState("none");
  const [subtitleApiModel, setSubtitleApiModel] = useState("gpt-4o-transcribe-diarize");
  const [maxIterations, setMaxIterations] = useState(20);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [showPreview, setShowPreview] = useState(false);

  const selectedVideo = videos.find((v) => v.path === videoPath);

  useEffect(() => {
    fetchVideos()
      .then((list) => {
        setVideos(list);
        if (list.length > 0) {
          setVideoPath(list[0].path);
          onVideoChange?.(list[0].path);
        }
      })
      .catch((err) => setLoadError(err.message));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSubmit = () => {
    if (videoPath.trim() && question.trim()) {
      onSubmit({
        videoPath: videoPath.trim(),
        question: question.trim(),
        subtitleMode,
        subtitleApiModel,
        maxIterations,
      });
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      handleSubmit();
    }
  };

  const selectClass =
    "w-full border border-hud-border bg-[#0a0a0a] px-2 py-2 text-sm text-foreground focus:border-hud-border-active focus:outline-none disabled:opacity-40 appearance-none";
  const labelClass =
    "block text-xs font-bold uppercase tracking-[0.15em] text-hud-dim mb-1";
  const inputClass =
    "w-full border border-hud-border bg-[#0a0a0a] px-2 py-2 text-sm text-foreground placeholder:text-hud-dim focus:border-hud-border-active focus:outline-none disabled:opacity-40";

  return (
    <Panel title="QUERY INPUT">
      <div className="space-y-3" onKeyDown={handleKeyDown}>
        {/* Video selector */}
        <div>
          <label className={labelClass}>VIDEO</label>
          {loadError ? (
            <p className="text-sm text-hud-red">{loadError}</p>
          ) : (
            <div className="flex gap-2">
              <select
                value={videoPath}
                onChange={(e) => {
                  setVideoPath(e.target.value);
                  onVideoChange?.(e.target.value);
                }}
                disabled={disabled || videos.length === 0}
                className={`flex-1 ${selectClass}`}
              >
                {videos.length === 0 && <option value="">Loading…</option>}
                {videos.map((v) => (
                  <option key={v.path} value={v.path}>{v.path}</option>
                ))}
              </select>
              <button
                type="button"
                onClick={() => setShowPreview(true)}
                disabled={!videoPath}
                title="Preview video & transcript"
                className="shrink-0 border border-hud-border px-2 py-2 text-hud-dim transition-colors hover:border-hud-border-active hover:text-foreground disabled:opacity-30"
              >
                <Eye size={16} />
              </button>
            </div>
          )}
          {/* Transcript status */}
          {selectedVideo && (
            <p className={`mt-1 text-[10px] uppercase tracking-[0.1em] ${
              selectedVideo.hasTranscript ? "text-hud-green" : "text-hud-dim"
            }`}>
              {selectedVideo.hasTranscript
                ? "● transcript loaded — will be used automatically"
                : "○ no transcript sidecar found"}
            </p>
          )}
        </div>

        {/* Question */}
        <div>
          <label className={labelClass}>QUESTION</label>
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="What is happening in the video?"
            disabled={disabled}
            className={inputClass}
          />
        </div>

        {/* Control knobs row */}
        <div className="grid grid-cols-3 gap-2">
          <div>
            <label className={labelClass}>SUBTITLE MODE</label>
            <select
              value={subtitleMode}
              onChange={(e) => setSubtitleMode(e.target.value)}
              disabled={disabled}
              className={selectClass}
            >
              {SUBTITLE_MODES.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>
          <div>
            <label className={labelClass}>SUBTITLE MODEL</label>
            <select
              value={subtitleApiModel}
              onChange={(e) => setSubtitleApiModel(e.target.value)}
              disabled={disabled || subtitleMode === "none" || subtitleMode === "local"}
              className={selectClass}
            >
              {SUBTITLE_API_MODELS.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>
          <div>
            <label className={labelClass}>MAX ITERATIONS</label>
            <input
              type="number"
              min={1}
              max={50}
              value={maxIterations}
              onChange={(e) => setMaxIterations(Number(e.target.value) || 20)}
              disabled={disabled}
              className={inputClass}
            />
          </div>
        </div>

        {/* Execute */}
        <button
          onClick={handleSubmit}
          disabled={disabled || !videoPath.trim() || !question.trim()}
          className="w-full border border-foreground bg-transparent px-3 py-2 text-xs font-bold uppercase tracking-[0.2em] text-foreground transition-colors hover:bg-foreground hover:text-background disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:bg-transparent disabled:hover:text-foreground"
        >
          {disabled ? "EXECUTING..." : "EXECUTE"}
        </button>
      </div>

      {showPreview && videoPath && (
        <VideoPreviewModal
          videoPath={videoPath}
          onClose={() => setShowPreview(false)}
        />
      )}
    </Panel>
  );
}
