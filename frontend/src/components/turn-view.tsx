"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Source } from "@/lib/api";
import { citedSources, type Turn } from "@/lib/turns";

export function TurnView({ turn }: { turn: Turn }) {
  const [open, setOpen] = useState<number | null>(null);
  const numbered = citedSources(turn);
  const numberOf = new Map(numbered.map(({ source, n }) => [source.id, n]));
  const markdown = withMarkers(turn, numberOf);
  const last = turn.retrievals.at(-1);
  const writing = turn.status === "running";

  return (
    <article className="flex flex-col gap-4">
      <p className="self-end rounded-2xl rounded-br-sm bg-surface-2 px-4 py-2 text-[15px]">
        {turn.question}
      </p>

      {turn.retrievals.length > 0 && (
        <details className="rounded-lg border border-border bg-surface px-4 py-2 text-xs text-muted">
          <summary className="cursor-pointer select-none">
            Searched {plural(last?.queries.length ?? 0, "query", "queries")} ·{" "}
            {plural(last?.candidates ?? 0, "candidate")} →{" "}
            {plural(last?.passages.length ?? 0, "passage")}
            {last?.confidence != null && ` · top relevance ${(last.confidence * 100).toFixed(0)}%`}
            {turn.retrievals.length > 1 && ` · ${turn.retrievals.length} rounds`}
          </summary>
          <div className="mt-2 flex flex-col gap-2 pb-1">
            {turn.retrievals.map((r) => (
              <div key={r.round}>
                <p className="mb-1 font-medium text-fg">Round {r.round}</p>
                <ul className="flex flex-wrap gap-1">
                  {r.queries.map((q) => (
                    <li key={q} className="rounded bg-surface-2 px-1.5 py-0.5 font-mono">
                      {q}
                    </li>
                  ))}
                </ul>
                <p className="mt-1">
                  {Object.entries(r.timings_ms)
                    .map(([k, v]) => `${k} ${v} ms`)
                    .join(" · ")}
                </p>
              </div>
            ))}
          </div>
        </details>
      )}

      {writing && !turn.blocks.length && (
        <p className="flex items-center gap-2 text-sm text-muted" aria-live="polite">
          <span className="size-1.5 animate-pulse rounded-full bg-accent" />
          {turn.retrievals.length ? "Reading sources…" : "Searching your documents…"}
        </p>
      )}

      {markdown && (
        <div
          className={`prose-answer text-[15px] leading-relaxed ${turn.status === "failed" ? "text-danger" : ""}`}
        >
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              a: ({ href, children }) => {
                const n = href?.startsWith("#cite-") ? Number(href.slice(6)) : null;
                if (n == null) return <a href={href}>{children}</a>;
                return (
                  <button
                    onClick={() => setOpen(open === n ? null : n)}
                    className="mx-0.5 inline-flex h-4 min-w-4 items-center justify-center rounded bg-accent-soft px-1 align-super font-mono text-[10px] text-accent"
                    aria-label={`Source ${n}`}
                  >
                    {n}
                  </button>
                );
              },
            }}
          >
            {markdown}
          </ReactMarkdown>
        </div>
      )}
      {turn.error && <p className="text-sm text-danger">{turn.error}</p>}

      {!writing && numbered.length > 0 && turn.status !== "chitchat" && (
        <section className="flex flex-col gap-2">
          <h3 className="text-xs font-medium uppercase tracking-wide text-muted">
            {turn.status === "no_evidence" ? "Closest passages" : "Sources"}
          </h3>
          {numbered.map(({ source, n }) => (
            <SourceCard
              key={source.id}
              n={n}
              source={source}
              quotes={turn.citations.filter((c) => c.passage === source.id).map((c) => c.text)}
              open={open === n}
              onToggle={() => setOpen(open === n ? null : n)}
            />
          ))}
        </section>
      )}
    </article>
  );
}

function SourceCard(props: {
  n: number;
  source: Source;
  quotes: string[];
  open: boolean;
  onToggle: () => void;
}) {
  const { n, source, quotes, open, onToggle } = props;
  return (
    <div className={`rounded-lg border bg-surface ${open ? "border-accent" : "border-border"}`}>
      <button onClick={onToggle} className="flex w-full items-start gap-3 px-3 py-2 text-left">
        <span className="mt-0.5 font-mono text-xs text-accent">{n}</span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-medium">{source.title}</span>
          <span className="block truncate text-xs text-muted">{source.location || "—"}</span>
        </span>
        {source.relevance != null && (
          <span className="font-mono text-xs text-muted">{(source.relevance * 100).toFixed(0)}%</span>
        )}
      </button>
      {open && (
        <p className="border-t border-border px-3 py-2 text-sm leading-relaxed text-muted">
          <Highlighted text={source.text} quotes={quotes} />
        </p>
      )}
    </div>
  );
}

function Highlighted({ text, quotes }: { text: string; quotes: string[] }) {
  // Match quotes whitespace-insensitively: citations may differ in line breaks.
  const ranges = quotes
    .map((q) => q.trim())
    .filter(Boolean)
    .flatMap((q) => {
      const pattern = q
        .split(/\s+/)
        .map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
        .join("\\s+");
      const match = new RegExp(pattern).exec(text);
      return match ? [[match.index, match.index + match[0].length] as const] : [];
    })
    .sort((a, b) => a[0] - b[0]);
  const parts: React.ReactNode[] = [];
  let cursor = 0;
  ranges.forEach(([start, end], i) => {
    if (start < cursor) return;
    parts.push(text.slice(cursor, start));
    parts.push(
      <mark key={i} className="cite text-fg">
        {text.slice(start, end)}
      </mark>,
    );
    cursor = end;
  });
  parts.push(text.slice(cursor));
  return <>{parts}</>;
}

/** Append citation links to each text block that cites something, then join. */
function withMarkers(turn: Turn, numberOf: Map<number, number>): string {
  return turn.blocks
    .map((text, block) => {
      const ns = [
        ...new Set(
          turn.citations
            .filter((c) => c.block === block)
            .map((c) => numberOf.get(c.passage))
            .filter((n): n is number => n != null),
        ),
      ];
      if (!ns.length) return text;
      const trailing = text.match(/\s*$/)?.[0] ?? "";
      const markers = ns.map((n) => `[${n}](#cite-${n})`).join("");
      return text.slice(0, text.length - trailing.length) + markers + trailing;
    })
    .join("");
}

function plural(n: number, one: string, many = `${one}s`): string {
  return `${n} ${n === 1 ? one : many}`;
}
