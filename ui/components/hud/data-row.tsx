import { cn } from "@/lib/utils";

interface DataRowProps {
  label: string;
  value: string | number;
  className?: string;
  valueClassName?: string;
}

export function DataRow({
  label,
  value,
  className,
  valueClassName,
}: DataRowProps) {
  return (
    <div className={cn("flex items-baseline justify-between gap-4", className)}>
      <span className="text-[12px] font-bold uppercase tracking-[0.15em] text-hud-dim shrink-0">
        {label}
      </span>
      <span
        className={cn(
          "text-xs tabular-nums text-right",
          valueClassName
        )}
      >
        {value}
      </span>
    </div>
  );
}
