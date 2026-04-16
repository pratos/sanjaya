"use client";

interface AnswerRendererProps {
  data: Record<string, unknown> | null;
  onSeek?: (seconds: number) => void;
}

type Row = Record<string, unknown>;

/**
 * Parse a timestamp string like "00:02:13", "00:02:13-00:02:17",
 * "02:13", "2:13.5", "132.5s" into seconds.
 * Returns the START time (before any dash) or null.
 */
function normalizeTimestampText(value: string): string {
  return value
    .replace(/[‐‑‒–—―−]/g, "-")
    .replace(/(\d),(\d)/g, "$1.$2")
    .trim();
}

function parseTimestamp(value: string): number | null {
  if (typeof value !== "string") return null;

  // Take only the start part if it's a range.
  const part = normalizeTimestampText(value).split(/\s*-\s*/)[0]?.trim() ?? "";

  // HH:MM:SS(.ms)
  const hmsMatch = part.match(/^(\d{1,2}):(\d{1,2}):(\d{1,2}(?:\.\d+)?)$/);
  if (hmsMatch) {
    return (
      parseInt(hmsMatch[1], 10) * 3600 +
      parseInt(hmsMatch[2], 10) * 60 +
      parseFloat(hmsMatch[3])
    );
  }

  // MM:SS(.ms)
  const msMatch = part.match(/^(\d{1,3}):(\d{1,2}(?:\.\d+)?)$/);
  if (msMatch) {
    return parseInt(msMatch[1], 10) * 60 + parseFloat(msMatch[2]);
  }

  // Plain seconds like "132.5", "132.5s", "132sec".
  // Avoid plain integers like "2026" to prevent false positives on years/IDs.
  const secMatch = part.match(/^(\d+(?:\.\d+)?)(?:\s*(s|sec|secs|second|seconds))?$/i);
  if (secMatch) {
    const hasUnit = Boolean(secMatch[2]);
    const hasDecimal = secMatch[1].includes(".");
    if (hasUnit || hasDecimal) return parseFloat(secMatch[1]);
  }

  return null;
}

function extractSeekSeconds(value: string): number | null {
  const normalized = normalizeTimestampText(value);

  const direct = parseTimestamp(normalized);
  if (direct != null) return direct;

  const candidates = normalized.match(
    /\d{1,3}:\d{1,2}(?:\.\d+)?|\d{1,2}:\d{1,2}:\d{1,2}(?:\.\d+)?|\d+(?:\.\d+)?\s*(?:s|sec|secs|second|seconds)\b|\d+\.\d+/gi
  ) ?? [];

  for (const candidate of candidates) {
    const parsed = parseTimestamp(candidate);
    if (parsed != null) return parsed;
  }

  return null;
}

/** Check if a column name looks like it holds timestamps. */
function isTimestampColumn(colName: string): boolean {
  const lower = colName.toLowerCase();
  return lower.includes("timestamp") || lower.includes("time") || lower === "start" || lower === "end";
}

function TimestampCell({
  value,
  onSeek,
}: {
  value: string;
  onSeek?: (seconds: number) => void;
}) {
  const seconds = extractSeekSeconds(value);
  if (seconds == null || !onSeek) {
    return <>{value}</>;
  }
  return (
    <button
      type="button"
      onClick={(e) => {
        e.stopPropagation();
        onSeek(seconds);
      }}
      className="text-hud-cyan hover:text-hud-green underline underline-offset-2 decoration-hud-cyan/40 hover:decoration-hud-green cursor-pointer transition-colors"
      title={`Seek to ${value}`}
    >
      {value}
    </button>
  );
}

export function AnswerRenderer({ data, onSeek }: AnswerRendererProps) {
  if (!data) {
    return <span className="text-[13px] text-hud-dim italic">No structured data</span>;
  }

  const sections: React.ReactNode[] = [];

  // Summary is always shown separately, skip it here
  const skip = new Set(["summary", "answer", "evidence"]);

  for (const [key, value] of Object.entries(data)) {
    if (skip.has(key)) continue;
    if (Array.isArray(value) && value.length > 0 && typeof value[0] === "object") {
      sections.push(
        <ListSection key={key} label={formatLabel(key)} items={value as Row[]} onSeek={onSeek} />
      );
    } else if (Array.isArray(value) && value.length > 0) {
      sections.push(
        <SimpleList key={key} label={formatLabel(key)} items={value as string[]} onSeek={onSeek} />
      );
    } else if (typeof value === "object" && value !== null) {
      sections.push(
        <ObjectSection key={key} label={formatLabel(key)} obj={value as Row} onSeek={onSeek} />
      );
    }
  }

  // Evidence section last
  const evidence = data.evidence;
  if (Array.isArray(evidence) && evidence.length > 0) {
    if (typeof evidence[0] === "string") {
      sections.push(
        <SimpleList key="evidence" label="Evidence" items={evidence as string[]} onSeek={onSeek} />
      );
    } else {
      sections.push(
        <ListSection key="evidence" label="Evidence" items={evidence as Row[]} onSeek={onSeek} />
      );
    }
  }

  if (sections.length === 0) {
    return (
      <pre className="text-[12px] text-hud-dim whitespace-pre-wrap overflow-x-auto">
        {JSON.stringify(data, null, 2)}
      </pre>
    );
  }

  return <div className="space-y-4">{sections}</div>;
}

function formatLabel(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/([A-Z])/g, " $1")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .trim();
}

function ListSection({ label, items, onSeek }: { label: string; items: Row[]; onSeek?: (s: number) => void }) {
  if (items.length === 0) return null;
  const columns = Object.keys(items[0]).filter(
    (k) => typeof items[0][k] !== "object" || items[0][k] === null
  );
  // Also include array columns as joined strings
  const arrayColumns = Object.keys(items[0]).filter(
    (k) => Array.isArray(items[0][k])
  );

  // Detect which columns are timestamps
  const timestampCols = new Set(
    [...columns, ...arrayColumns].filter((col) => {
      if (isTimestampColumn(col)) return true;
      // Also check if first row value looks like a timestamp
      const val = String(items[0][col] ?? "");
      return parseTimestamp(val) != null && val.includes(":");
    })
  );

  return (
    <div>
      <span className="block text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-1">
        {label} ({items.length})
      </span>
      <div className="overflow-x-auto">
        <table className="w-full text-[12px] table-fixed">
          <thead>
            <tr className="border-b border-hud-border">
              {[...columns, ...arrayColumns].map((col) => (
                <th
                  key={col}
                  className="px-2 py-1 text-left text-[13px] font-bold uppercase tracking-wider text-hud-dim"
                >
                  {formatLabel(col)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {items.map((item, i) => (
              <tr key={i} className="border-b border-hud-border/50">
                {columns.map((col) => {
                  const val = String(item[col] ?? "\u2014");
                  const isPureTimestamp = timestampCols.has(col) && extractSeekSeconds(val) != null;
                  return (
                    <td key={col} className="px-2 py-1 text-hud-label max-w-[400px] whitespace-normal break-words">
                      {isPureTimestamp ? (
                        <TimestampCell value={val} onSeek={onSeek} />
                      ) : (
                        <InlineTimestamps text={val} onSeek={onSeek} />
                      )}
                    </td>
                  );
                })}
                {arrayColumns.map((col) => {
                  const joined = (item[col] as string[])?.join(", ") ?? "\u2014";
                  return (
                    <td key={col} className="px-2 py-1 text-hud-label max-w-[300px] whitespace-normal break-words">
                      <InlineTimestamps text={joined} onSeek={onSeek} />
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/**
 * Render a string with inline timestamps as clickable links.
 * Matches patterns like [00:22], [89.92-91.88], (02:13), 00:44, etc.
 */
export function InlineTimestamps({ text, onSeek }: { text: string; onSeek?: (s: number) => void }) {
  if (!onSeek) return <>{text}</>;

  // Match timestamps in brackets/parens and standalone clock-like values.
  // Examples: [00:22], (89.92-91.88), 00:44, 01:02:33.5, 00:12-00:20
  const parts = text.split(
    /(\[[^\]]+\]|\([^\)]+\)|\b\d{1,3}:\d{1,2}(?:[\.,]\d+)?(?:\s*[-‐‑‒–—―−]\s*\d{1,3}:\d{1,2}(?:[\.,]\d+)?)?\b|\b\d{1,2}:\d{1,2}:\d{1,2}(?:[\.,]\d+)?(?:\s*[-‐‑‒–—―−]\s*\d{1,2}:\d{1,2}:\d{1,2}(?:[\.,]\d+)?)?\b|\b\d+(?:[\.,]\d+)?\s*(?:s|sec|secs|second|seconds)(?:\s*[-‐‑‒–—―−]\s*\d+(?:[\.,]\d+)?\s*(?:s|sec|secs|second|seconds)?)?\b|\b\d+(?:[\.,]\d+)(?:\s*[-‐‑‒–—―−]\s*\d+(?:[\.,]\d+))?\b)/gi
  );

  return (
    <>
      {parts.map((part, i) => {
        const stripped = part.replace(/^[\[(]|[\])]{1}$/g, "").trim();
        const start = normalizeTimestampText(stripped).split(/\s*[-‐‑‒–—―−,]\s*/)[0].trim();
        const seconds = extractSeekSeconds(start);
        if (seconds != null) {
          return (
            <button
              key={i}
              type="button"
              onClick={(e) => { e.stopPropagation(); onSeek(seconds); }}
              className="text-hud-cyan hover:text-hud-green underline underline-offset-2 decoration-hud-cyan/40 hover:decoration-hud-green cursor-pointer transition-colors"
              title={`Seek to ${start}`}
            >
              {part}
            </button>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}

function SimpleList({ label, items, onSeek }: { label: string; items: string[]; onSeek?: (s: number) => void }) {
  return (
    <div>
      <span className="block text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-1">
        {label} ({items.length})
      </span>
      <ul className="space-y-0.5">
        {items.map((item, i) => (
          <li key={i} className="text-[12px] text-hud-label pl-2 border-l border-hud-border/50">
            <InlineTimestamps text={item} onSeek={onSeek} />
          </li>
        ))}
      </ul>
    </div>
  );
}

function ObjectSection({ label, obj, onSeek }: { label: string; obj: Row; onSeek?: (s: number) => void }) {
  return (
    <div>
      <span className="block text-[13px] font-bold uppercase tracking-[0.15em] text-hud-dim mb-1">
        {label}
      </span>
      <div className="space-y-0.5">
        {Object.entries(obj).map(([k, v]) => {
          const text = String(v);
          return (
            <div key={k} className="flex gap-2 text-[12px]">
              <span className="text-hud-dim shrink-0">{formatLabel(k)}:</span>
              <span className="text-hud-label"><InlineTimestamps text={text} onSeek={onSeek} /></span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
