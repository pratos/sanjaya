import { cn } from "@/lib/utils";

interface PanelProps {
  title: string;
  status?: "idle" | "active" | "error" | "success";
  children: React.ReactNode;
  className?: string;
}

const statusColors: Record<string, string> = {
  idle: "bg-hud-dim",
  active: "bg-hud-amber animate-pulse",
  error: "bg-hud-red",
  success: "bg-hud-green",
};

export function Panel({ title, status, children, className }: PanelProps) {
  return (
    <div
      className={cn(
        "border border-hud-border bg-hud-panel flex flex-col overflow-hidden",
        className
      )}
    >
      <div className="flex items-center gap-2 border-b border-hud-border px-3 py-2">
        {status && (
          <span
            className={cn("h-1.5 w-1.5 shrink-0", statusColors[status])}
          />
        )}
        <span className="text-[10px] font-bold uppercase tracking-[0.15em] text-hud-label">
          {title}
        </span>
      </div>
      <div className="flex-1 min-h-0 overflow-auto p-3">{children}</div>
    </div>
  );
}
