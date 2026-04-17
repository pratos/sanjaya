"use client";

import { Children, isValidElement, useEffect, useMemo, useState, type ReactNode } from "react";
import Link from "next/link";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { Panel } from "@/components/hud/panel";
import { YouTubePlayer } from "@/components/youtube-player";

type Frontmatter = {
  title?: string;
  description?: string;
  date?: string;
  author?: string;
  tags?: string;
};

type ParsedArticle = {
  frontmatter: Frontmatter;
  body: string;
};

const CORRECT_PIN = "kc51BoN62fnzpSs";
const ARTICLE_PATH = "/blog/building-sanjaya.md";

function extractYouTubeId(url?: string): string | null {
  if (!url) return null;
  const m = url.match(
    /(?:youtu\.be\/|youtube\.com\/(?:watch\?v=|embed\/|shorts\/))([A-Za-z0-9_-]{11})/
  );
  return m?.[1] ?? null;
}

function parseFrontmatter(markdown: string): ParsedArticle {
  const match = markdown.match(/^---\n([\s\S]*?)\n---\n?/);
  if (!match) {
    return { frontmatter: {}, body: markdown };
  }

  const metaBlock = match[1];
  const frontmatter: Frontmatter = {};

  for (const line of metaBlock.split("\n")) {
    const idx = line.indexOf(":");
    if (idx <= 0) continue;
    const key = line.slice(0, idx).trim();
    let value = line.slice(idx + 1).trim();

    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }

    if (key === "title") frontmatter.title = value;
    if (key === "description") frontmatter.description = value;
    if (key === "date") frontmatter.date = value;
    if (key === "author") frontmatter.author = value;
    if (key === "tags") frontmatter.tags = value;
  }

  return { frontmatter, body: markdown.slice(match[0].length) };
}

function firstImageChild(children: ReactNode): { alt?: string } | null {
  const nodes = Children.toArray(children);
  for (const node of nodes) {
    if (!isValidElement<{ alt?: string; src?: string }>(node)) continue;
    if (typeof node.props?.src === "string") {
      return { alt: node.props.alt };
    }
  }
  return null;
}

export function BuildingSanjayaArticlePage() {
  const [pin, setPin] = useState("");
  const [pinError, setPinError] = useState<string | null>(null);
  const [unlocked, setUnlocked] = useState(false);
  const [article, setArticle] = useState<ParsedArticle | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    if (!unlocked || article) return;

    fetch(ARTICLE_PATH, { cache: "no-store" })
      .then(async (res) => {
        if (!res.ok) throw new Error(`Failed to load markdown (${res.status})`);
        const raw = await res.text();
        setArticle(parseFrontmatter(raw));
      })
      .catch((err: unknown) => {
        setLoadError(err instanceof Error ? err.message : "Unable to load article.");
      });
  }, [article, unlocked]);

  const markdownComponents = useMemo<Components>(
    () => ({
      h1: ({ children }) => (
        <h1 className="mt-6 mb-3 text-2xl font-bold text-foreground tracking-[0.02em]">{children}</h1>
      ),
      h2: ({ children }) => (
        <h2 className="mt-8 mb-3 border-b border-hud-border pb-2 text-xl font-bold uppercase tracking-[0.08em] text-hud-label">
          {children}
        </h2>
      ),
      h3: ({ children }) => (
        <h3 className="mt-6 mb-2 text-lg font-bold text-hud-cyan">{children}</h3>
      ),
      p: ({ children }) => <p className="my-3 text-sm leading-7 text-foreground/90">{children}</p>,
      blockquote: ({ children }) => (
        <blockquote className="my-4 border-l-2 border-hud-blue bg-hud-panel/40 px-4 py-2 text-sm italic text-hud-label">
          {children}
        </blockquote>
      ),
      ul: ({ children }) => <ul className="my-3 list-disc space-y-2 pl-6 text-sm text-foreground/90">{children}</ul>,
      ol: ({ children }) => <ol className="my-3 list-decimal space-y-2 pl-6 text-sm text-foreground/90">{children}</ol>,
      li: ({ children }) => <li className="leading-7">{children}</li>,
      hr: () => <hr className="my-8 border-hud-border" />,
      a: ({ href, children }) => {
        const ytId = extractYouTubeId(href);
        const image = firstImageChild(children);

        if (ytId && image) {
          return (
            <div className="my-6 space-y-2">
              <YouTubePlayer videoId={ytId} className="w-full overflow-hidden border border-hud-border" />
              <p className="text-[11px] uppercase tracking-[0.12em] text-hud-dim">
                {(image.alt || "Video").replace(/^Watch:\s*/i, "")}
              </p>
            </div>
          );
        }

        if (ytId) {
          return (
            <div className="my-6 space-y-2">
              <YouTubePlayer videoId={ytId} className="w-full overflow-hidden border border-hud-border" />
              <p className="text-[11px] uppercase tracking-[0.12em] text-hud-dim">Video clip</p>
            </div>
          );
        }

        if (!href) return <>{children}</>;

        const isInternal = href.startsWith("/");
        if (isInternal) {
          return (
            <Link href={href} className="text-hud-cyan underline underline-offset-2 hover:text-foreground">
              {children}
            </Link>
          );
        }

        return (
          <a
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className="text-hud-cyan underline underline-offset-2 hover:text-foreground"
          >
            {children}
          </a>
        );
      },
      img: ({ alt, src }) => (
        <img src={src ?? ""} alt={alt ?? ""} className="my-4 w-full border border-hud-border" />
      ),
      code: ({ className, children }) => {
        const isBlock = Boolean(className);
        if (!isBlock) {
          return <code className="rounded bg-[#151515] px-1.5 py-0.5 text-hud-amber">{children}</code>;
        }
        return (
          <pre className="my-4 overflow-x-auto border border-hud-border bg-[#101010] p-3">
            <code className={className}>{children}</code>
          </pre>
        );
      },
      table: ({ children }) => (
        <div className="my-4 overflow-x-auto border border-hud-border">
          <table className="w-full border-collapse text-left text-sm">{children}</table>
        </div>
      ),
      thead: ({ children }) => <thead className="bg-[#121212] text-hud-label">{children}</thead>,
      th: ({ children }) => <th className="border border-hud-border px-3 py-2 text-xs uppercase tracking-[0.08em]">{children}</th>,
      td: ({ children }) => <td className="border border-hud-border px-3 py-2 align-top text-foreground/90">{children}</td>,
    }),
    []
  );

  const handleUnlock = () => {
    if (pin.trim() !== CORRECT_PIN) {
      setPinError("Incorrect PIN");
      setPin("");
      return;
    }
    setPinError(null);
    setUnlocked(true);
  };

  return (
    <main className="flex-1 p-4">
      <div className="mx-auto w-full max-w-5xl">
        {!unlocked ? (
          <div className="mx-auto mt-16 max-w-xl">
            <Panel title="access /building-sanjaya" status="active">
              <div className="space-y-4">
                <p className="text-xs uppercase tracking-[0.15em] text-hud-label">Enter PIN to unlock the article</p>
                <input
                  value={pin}
                  onChange={(e) => setPin(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleUnlock()}
                  type="password"
                  className="w-full border border-hud-border bg-background px-3 py-2 text-sm text-foreground outline-none focus:border-hud-cyan"
                  placeholder="PIN"
                />
                <button
                  onClick={handleUnlock}
                  className="border border-hud-border bg-hud-panel px-4 py-2 text-xs font-bold uppercase tracking-[0.15em] text-foreground hover:border-hud-cyan hover:text-hud-cyan"
                >
                  Enter
                </button>
                {pinError ? <p className="text-xs text-hud-red">{pinError}</p> : null}
              </div>
            </Panel>
          </div>
        ) : (
          <Panel title="building-sanjaya.md" status="success" className="min-h-[70vh]">
            {!article && !loadError ? (
              <p className="text-xs uppercase tracking-[0.15em] text-hud-label animate-pulse">Loading article…</p>
            ) : null}

            {loadError ? <p className="text-sm text-hud-red">{loadError}</p> : null}

            {article ? (
              <article className="max-w-none">
                <header className="mb-6 border-b border-hud-border pb-4">
                  <h1 className="text-2xl font-bold text-foreground">{article.frontmatter.title ?? "Building Sanjaya"}</h1>
                  {article.frontmatter.description ? (
                    <p className="mt-2 text-sm text-hud-label">{article.frontmatter.description}</p>
                  ) : null}
                  <div className="mt-3 flex flex-wrap gap-3 text-[11px] uppercase tracking-[0.12em] text-hud-dim">
                    {article.frontmatter.date ? <span>{article.frontmatter.date}</span> : null}
                    {article.frontmatter.author ? <span>{article.frontmatter.author}</span> : null}
                    <Link href="/" className="text-hud-cyan hover:text-foreground">
                      Back to dashboard
                    </Link>
                  </div>
                </header>

                <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                  {article.body}
                </ReactMarkdown>
              </article>
            ) : null}
          </Panel>
        )}
      </div>
    </main>
  );
}
