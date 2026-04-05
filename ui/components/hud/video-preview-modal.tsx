"use client";

import { useEffect, useRef, useState } from "react";
import { X } from "lucide-react";
import {
  fetchTranscript,
  videoStreamUrl,
  type TranscriptSegment,
} from "@/lib/api";

interface VideoPreviewModalProps {
  videoPath: string;
  onClose: () => void;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function VideoPreviewModal({
  videoPath,
  onClose,
}: VideoPreviewModalProps) {
  const [segments, setSegments] = useState<TranscriptSegment[]>([]);
  const [loading, setLoading] = useState(true);
  const [currentTime, setCurrentTime] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const activeSegRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setLoading(true);
    fetchTranscript(videoPath)
      .then(setSegments)
      .finally(() => setLoading(false));
  }, [videoPath]);

  // Auto-scroll to active segment
  useEffect(() => {
    activeSegRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "nearest",
    });
  }, [currentTime]);

  // Close on Escape
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const seekTo = (time: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      videoRef.current.play();
    }
  };

  const activeIndex = segments.findIndex(
    (s) => currentTime >= s.start && currentTime < s.end
  );

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="relative flex h-[85vh] w-[90vw] max-w-6xl flex-col border border-hud-border bg-[#0a0a0a]">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-hud-border px-4 py-2">
          <h2 className="text-xs font-bold uppercase tracking-[0.15em] text-hud-dim">
            VIDEO PREVIEW — {videoPath}
          </h2>
          <button
            onClick={onClose}
            className="text-hud-dim transition-colors hover:text-foreground"
          >
            <X size={16} />
          </button>
        </div>

        {/* Body */}
        <div className="flex flex-1 overflow-hidden">
          {/* Video player */}
          <div className="flex w-1/2 flex-col border-r border-hud-border">
            <div className="flex-1 flex items-center justify-center bg-black p-4">
              <video
                ref={videoRef}
                src={videoStreamUrl(videoPath)}
                controls
                className="max-h-full max-w-full"
                onTimeUpdate={() =>
                  setCurrentTime(videoRef.current?.currentTime ?? 0)
                }
              />
            </div>
          </div>

          {/* Transcript */}
          <div className="flex w-1/2 flex-col">
            <div className="border-b border-hud-border px-4 py-2">
              <h3 className="text-xs font-bold uppercase tracking-[0.15em] text-hud-dim">
                AUDIO TRANSCRIPT
                {segments.length > 0 && (
                  <span className="ml-2 text-hud-green">
                    {segments.length} segments
                  </span>
                )}
              </h3>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-1">
              {loading && (
                <p className="text-sm text-hud-dim">Loading transcript…</p>
              )}
              {!loading && segments.length === 0 && (
                <p className="text-sm text-hud-dim">
                  No transcript available for this video.
                </p>
              )}
              {segments.map((seg, i) => {
                const isActive = i === activeIndex;
                return (
                  <div
                    key={i}
                    ref={isActive ? activeSegRef : undefined}
                    onClick={() => seekTo(seg.start)}
                    className={`flex cursor-pointer gap-3 px-2 py-1.5 text-sm transition-colors ${
                      isActive
                        ? "bg-[#1a1a1a] text-hud-amber border-l-2 border-hud-amber"
                        : "text-foreground/80 hover:bg-[#141414] border-l-2 border-transparent"
                    }`}
                  >
                    <span className="shrink-0 font-mono text-xs text-hud-dim w-[5ch] text-right">
                      {formatTime(seg.start)}
                    </span>
                    <span>{seg.text}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
