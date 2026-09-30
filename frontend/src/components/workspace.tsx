"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  deleteDocument,
  listDocuments,
  streamChat,
  uploadDocuments,
  type DocumentInfo,
} from "@/lib/api";
import { applyEvent, newTurn, type Turn } from "@/lib/turns";
import { Library } from "./library";
import { TurnView } from "./turn-view";

export function Workspace() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [apiError, setApiError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [turns, setTurns] = useState<Turn[]>([]);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const refresh = useCallback(async () => {
    try {
      setDocuments(await listDocuments());
      setApiError(null);
    } catch (err) {
      setApiError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    listDocuments(controller.signal)
      .then(setDocuments)
      .catch((err: unknown) => {
        if (!controller.signal.aborted) setApiError(String(err));
      });
    return () => controller.abort();
  }, []);

  // Poll while anything is still being indexed.
  const indexing = documents.some((d) => d.status === "processing");
  useEffect(() => {
    if (!indexing) return;
    const timer = setInterval(() => void refresh(), 1500);
    return () => clearInterval(timer);
  }, [indexing, refresh]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  const ask = useCallback(
    async (question: string) => {
      const text = question.trim();
      if (!text || busy) return;
      const turn = newTurn(text);
      setTurns((prev) => [...prev, turn]);
      setInput("");
      setBusy(true);
      const controller = new AbortController();
      abortRef.current = controller;
      const update = (fn: (t: Turn) => Turn) =>
        setTurns((prev) => prev.map((t) => (t.id === turn.id ? fn(t) : t)));
      try {
        const scope = selected.size ? [...selected] : null;
        for await (const event of streamChat(text, threadId, scope, controller.signal)) {
          if (event.event === "thread") setThreadId(event.thread_id);
          else update((t) => applyEvent(t, event));
        }
      } catch (err) {
        const message = controller.signal.aborted ? "Stopped." : `Connection error: ${String(err)}`;
        update((t) => ({ ...t, status: "failed", error: message }));
      } finally {
        update((t) =>
          t.status === "running" ? { ...t, status: "failed", error: t.error ?? "No answer received." } : t,
        );
        setBusy(false);
        abortRef.current = null;
      }
    },
    [busy, threadId, selected],
  );

  const ready = documents.filter((d) => d.status === "ready").length;

  return (
    <div className="flex h-full">
      <aside
        className={`${sidebarOpen ? "fixed inset-0 z-20 flex" : "hidden"} w-full flex-col border-r border-border bg-surface md:static md:flex md:w-80 md:shrink-0`}
      >
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <span className="text-sm font-medium">Library</span>
          <button className="text-sm text-muted md:hidden" onClick={() => setSidebarOpen(false)}>
            Close
          </button>
        </div>
        <Library
          documents={documents}
          error={apiError}
          selected={selected}
          onToggle={(id) =>
            setSelected((prev) => {
              const next = new Set(prev);
              if (next.has(id)) next.delete(id);
              else next.add(id);
              return next;
            })
          }
          onUpload={async (files) => {
            await uploadDocuments(files);
            await refresh();
          }}
          onDelete={(id) => {
            void deleteDocument(id).then(() => {
              setSelected((prev) => {
                const next = new Set(prev);
                next.delete(id);
                return next;
              });
              return refresh();
            });
          }}
        />
      </aside>

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-border px-4 py-3 md:px-8">
          <div className="flex items-center gap-3">
            <button
              className="rounded-md border border-border px-2 py-1 text-xs text-muted md:hidden"
              onClick={() => setSidebarOpen(true)}
            >
              Library
            </button>
            <h1 className="text-sm font-semibold tracking-tight">DocuChat</h1>
            <span className="rounded-full bg-surface-2 px-2 py-0.5 font-mono text-[11px] text-muted">
              {ready} indexed
            </span>
          </div>
          {turns.length > 0 && (
            <button
              onClick={() => {
                abortRef.current?.abort();
                setTurns([]);
                setThreadId(null);
              }}
              className="text-sm text-muted hover:text-fg"
            >
              New chat
            </button>
          )}
        </header>

        <div className="flex-1 overflow-y-auto">
          <div className="mx-auto flex max-w-3xl flex-col gap-10 px-4 py-8 md:px-8">
            {turns.length === 0 ? (
              <div className="flex flex-col gap-3 pt-10">
                <h2 className="text-2xl font-semibold tracking-tight">
                  Answers from your documents, with receipts.
                </h2>
                <p className="max-w-xl text-muted">
                  Hybrid search (dense, BM25 and SPLADE) finds candidate passages, a cross-encoder
                  reranks them, and Claude answers citing the exact sentences it used. If the
                  documents don&apos;t cover your question, it says so.
                </p>
                {ready === 0 && (
                  <p className="text-sm text-accent">Add a document in the library to get started.</p>
                )}
              </div>
            ) : (
              turns.map((turn) => <TurnView key={turn.id} turn={turn} />)
            )}
            <div ref={bottomRef} />
          </div>
        </div>

        <form
          className="border-t border-border bg-bg px-4 py-4 md:px-8"
          onSubmit={(e) => {
            e.preventDefault();
            void ask(input);
          }}
        >
          <div className="mx-auto flex max-w-3xl items-end gap-2 rounded-xl border border-border bg-surface p-2 focus-within:border-accent">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void ask(input);
                }
              }}
              rows={1}
              placeholder={ready ? "Ask about your documents…" : "Upload a document first…"}
              aria-label="Question"
              className="max-h-40 min-h-10 flex-1 resize-none bg-transparent px-2 py-2 text-[15px] outline-none placeholder:text-muted"
            />
            {busy ? (
              <button
                type="button"
                onClick={() => abortRef.current?.abort()}
                className="rounded-lg border border-border px-4 py-2 text-sm"
              >
                Stop
              </button>
            ) : (
              <button
                type="submit"
                disabled={!input.trim()}
                className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-bg disabled:opacity-40"
              >
                Ask
              </button>
            )}
          </div>
        </form>
      </main>
    </div>
  );
}
