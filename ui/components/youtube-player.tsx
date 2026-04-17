"use client";

import { useEffect, useImperativeHandle, useRef, forwardRef } from "react";

export interface YouTubePlayerHandle {
  seekTo: (seconds: number) => void;
}

interface YouTubePlayerProps {
  videoId: string;
  className?: string;
}

/* Global YT IFrame API loading — shared across all player instances. */
let apiLoadPromise: Promise<void> | null = null;

function ensureYTApi(): Promise<void> {
  if (apiLoadPromise) return apiLoadPromise;
  apiLoadPromise = new Promise<void>((resolve) => {
    if (typeof window !== "undefined" && (window as any).YT?.Player) {
      resolve();
      return;
    }
    const prev = (window as any).onYouTubeIframeAPIReady;
    (window as any).onYouTubeIframeAPIReady = () => {
      prev?.();
      resolve();
    };
    const tag = document.createElement("script");
    tag.src = "https://www.youtube.com/iframe_api";
    document.head.appendChild(tag);
  });
  return apiLoadPromise;
}

export const YouTubePlayer = forwardRef<YouTubePlayerHandle, YouTubePlayerProps>(
  function YouTubePlayer({ videoId, className }, ref) {
    const containerRef = useRef<HTMLDivElement>(null);
    const playerRef = useRef<any>(null);

    useImperativeHandle(ref, () => ({
      seekTo(seconds: number) {
        playerRef.current?.seekTo(seconds, true);
        playerRef.current?.playVideo();
      },
    }));

    useEffect(() => {
      const el = containerRef.current;
      if (!el) return;

      let destroyed = false;
      ensureYTApi().then(() => {
        if (destroyed || !containerRef.current) return;
        playerRef.current = new (window as any).YT.Player(containerRef.current, {
          videoId,
          width: "100%",
          height: "100%",
          playerVars: { modestbranding: 1, rel: 0 },
        });
      });

      return () => {
        destroyed = true;
        try { playerRef.current?.destroy(); } catch { /* noop */ }
        playerRef.current = null;
      };
    }, [videoId]);

    return (
      <div className={className} style={{ aspectRatio: "16/9" }}>
        <div ref={containerRef} style={{ width: "100%", height: "100%" }} />
      </div>
    );
  },
);
