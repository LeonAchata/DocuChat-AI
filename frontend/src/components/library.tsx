"use client";

import { useRef, useState } from "react";
import type { DocumentInfo } from "@/lib/api";

type Props = {
  documents: DocumentInfo[];
  error: string | null;
  selected: Set<string>;
  onToggle: (id: string) => void;
  onUpload: (files: File[]) => Promise<void>;
  onDelete: (id: string) => void;
};

const ACCEPT = ".pdf,.docx,.md,.markdown,.txt,.html,.htm";

export function Library({ documents, error, selected, onToggle, onUpload, onDelete }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const upload = async (files: File[]) => {
    if (!files.length) return;
    setUploading(true);
    setUploadError(null);
    try {
      await onUpload(files);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 p-3">
      <button
        type="button"
        onClick={() => input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          void upload(Array.from(e.dataTransfer.files));
        }}
        className={`rounded-lg border border-dashed px-4 py-5 text-center text-sm transition-colors ${
          dragging ? "border-accent bg-accent-soft" : "border-border hover:border-accent"
        }`}
      >
        <span className="font-medium">{uploading ? "Uploading…" : "Add documents"}</span>
        <span className="mt-1 block text-xs text-muted">PDF, DOCX, Markdown, HTML, TXT</span>
      </button>
      <input
        ref={input}
        type="file"
        multiple
        accept={ACCEPT}
        className="hidden"
        onChange={(e) => {
          void upload(Array.from(e.target.files ?? []));
          e.target.value = "";
        }}
      />
      {uploadError && <p className="text-xs text-danger">{uploadError}</p>}
      {error && <p className="text-xs text-danger">Could not reach the API: {error}</p>}

      <div className="flex items-center justify-between px-1 text-xs text-muted">
        <span>{documents.length} document{documents.length === 1 ? "" : "s"}</span>
        {selected.size > 0 && <span>Searching {selected.size} selected</span>}
      </div>

      <ul className="-mx-1 flex-1 overflow-y-auto">
        {documents.map((d) => (
          <li key={d.id} className="group flex items-start gap-2 rounded-md px-2 py-2 hover:bg-surface-2">
            <input
              type="checkbox"
              aria-label={`Limit search to ${d.title}`}
              checked={selected.has(d.id)}
              disabled={d.status !== "ready"}
              onChange={() => onToggle(d.id)}
              className="mt-1 accent-[var(--accent)]"
            />
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm" title={d.filename}>
                {d.title}
              </p>
              <p className="text-xs text-muted">
                {d.status === "ready" && (
                  <>
                    {d.pages ? `${d.pages} pages · ` : ""}
                    {d.chunk_count} chunks
                  </>
                )}
                {d.status === "processing" && <span className="animate-pulse">Indexing…</span>}
                {d.status === "failed" && <span className="text-danger">{d.error ?? "Failed"}</span>}
              </p>
            </div>
            <button
              onClick={() => onDelete(d.id)}
              aria-label={`Delete ${d.title}`}
              className="text-xs text-muted opacity-0 hover:text-danger group-hover:opacity-100 focus:opacity-100"
            >
              Remove
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
