"use client";

interface NarrativeNoteProps {
  title: string;
  text: string;
  improvement: string;
}

export function NarrativeNote({ title, text, improvement }: NarrativeNoteProps) {
  return (
    <div className="border-l-2 border-hud-cyan bg-hud-cyan/5 px-3 py-2 my-3">
      <div className="flex items-center gap-2 mb-1">
        <span className="text-[9px] font-bold uppercase tracking-[0.15em] text-hud-cyan">
          {title}
        </span>
        <span className="text-[9px] tabular-nums text-hud-green">
          {improvement}
        </span>
      </div>
      <p className="text-[11px] text-hud-label leading-relaxed">{text}</p>
    </div>
  );
}
