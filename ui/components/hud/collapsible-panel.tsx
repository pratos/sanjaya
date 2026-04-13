"use client";

import { useState } from "react";
import { ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

interface CollapsiblePanelProps {
  title: string;
  count?: number;
  status?: "idle" | "active" | "error" | "success";
  defaultOpen?: boolean;
  children: React.ReactNode;
  expandedHeight?: string;
}

const statusColors: Record<string, string> = {
  idle: "bg-hud-dim",
  active: "bg-hud-amber animate-pulse",
  error: "bg-hud-red",
  success: "bg-hud-green",
};

export function CollapsiblePanel({
  title,
  count,
  status,
  defaultOpen = false,
  children,
  expandedHeight = "h-72",
}: CollapsiblePanelProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="border border-hud-border bg-hud-panel">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-2 px-3 py-2 hover:bg-[#141414] transition-colors"
      >
        <ChevronRight
          size={12}
          className={cn(
            "shrink-0 text-hud-dim transition-transform duration-150",
            open && "rotate-90"
          )}
        />
        {status && (
          <span className={cn("h-1.5 w-1.5 shrink-0", statusColors[status])} />
        )}
        <span className="text-[12px] font-bold uppercase tracking-[0.15em] text-hud-label">
          {title}
        </span>
        {count != null && count > 0 && (
          <span className="text-[13px] tabular-nums text-hud-dim">{count}</span>
        )}
      </button>
      {open && (
        <div className={cn("border-t border-hud-border", expandedHeight)}>
          <div className="h-full overflow-auto p-3">{children}</div>
        </div>
      )}
    </div>
  );
}
